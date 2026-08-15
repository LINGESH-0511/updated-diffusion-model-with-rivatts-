"""
backend/audio2face/stream.py

Phase 2: Send audio + config to Audio2Face over an already-open
ProcessAudioStream() stream (from client.py).

Phase 6: Also supports sending audio as it streams in incrementally
(e.g. from a streaming TTS engine) via send_audio_stream(), instead of
requiring the full audio buffer up front.

Phase 7 (emotion wiring): both send_audio() and send_audio_stream() now
accept an optional `preferred_emotion` dict -- a per-call override for
which facial emotion A2F should blend in, sourced from
backend/emotion/classifier.py via bridge_server.py, instead of always
reading the static emotion_with_timecode_list baked into the YAML config.
This is what lets facial emotion track the classifier's per-sentence
output instead of relying solely on A2E's own (flat) inference from TTS
audio prosody. See _build_emotion_timecode_list() below.

Phase 8 (emotion vector fix): preferred_emotion is now always expanded
into a FULL 6-key vector before being sent to A2F. Previously a sparse
single-key dict (e.g. {"joy": 1.0}) was sent, which A2F may silently
ignore or clamp because its gRPC EmotionWithTimeCode expects all 6
emotion weights to be present. The full vector explicitly zeroes out all
non-target emotions so the intent is unambiguous.

Responsibilities:
    - Build and send AudioStreamHeader (face/blendshape/emotion config)
    - Chunk raw PCM16 mono audio and send as AudioWithEmotion messages
    - Send EndOfAudio to signal completion

Does NOT:
    - Open the gRPC channel (client.py's job)
    - Read WAV files from disk (wav_sender.py's job, Phase 4)
    - Generate TTS audio (chatterbox_streamer.py's job, Phase 6)
    - Read blendshapes/animation data back (receiver.py's job, Phase 3)
"""

import numpy as np

from nvidia_ace.a2f.v1_pb2 import (
    AudioWithEmotion,
    EmotionPostProcessingParameters,
    FaceParameters,
    BlendShapeParameters,
    EmotionParameters,
)
from nvidia_ace.audio.v1_pb2 import AudioHeader
from nvidia_ace.controller.v1_pb2 import AudioStream, AudioStreamHeader
from nvidia_ace.emotion_with_timecode.v1_pb2 import EmotionWithTimeCode

from backend.utils.logger import get_logger
from backend.utils.exceptions import Audio2FaceStreamError

logger = get_logger(__name__)

# Only 16-bit mono PCM is supported by Audio2Face-3D.
REQUIRED_BITS_PER_SAMPLE = 16
REQUIRED_CHANNEL_COUNT = 1
REQUIRED_AUDIO_FORMAT = AudioHeader.AUDIO_FORMAT_PCM

# A2F's real facial emotion set — exactly these 6 values, no more.
# Sending a full vector (all 6 keys, non-target ones at 0.0) instead of
# a sparse dict prevents A2F from silently ignoring the override because
# it received an incomplete / unexpected payload shape.
_ALL_A2F_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness"]


def _build_full_emotion_vector(preferred_emotion: dict[str, float]) -> dict[str, float]:
    """
    Expands a sparse preferred_emotion dict (e.g. {"joy": 1.0}) into a
    full 6-key vector with all non-target emotions explicitly set to 0.0.

    A2F's gRPC EmotionWithTimeCode proto expects weights for all 6
    emotions. Sending only the target key may cause the remaining weights
    to be interpreted as "unset" (potentially inherited from A2E's own
    audio-based inference) rather than "zero", which would dilute or
    entirely override the preferred emotion. The full vector makes the
    intent unambiguous: target emotion at full weight, everything else off.

    Example:
        {"joy": 1.0}  ->  {"anger": 0.0, "disgust": 0.0, "fear": 0.0,
                            "joy": 1.0, "neutral": 0.0, "sadness": 0.0}
    """
    full_vector = {e: 0.0 for e in _ALL_A2F_EMOTIONS}
    full_vector.update(preferred_emotion)  # overwrite only the target key(s)
    return full_vector


def _build_audio_stream_header(samplerate: int, bit_depth: int, channels: int, config: dict) -> AudioStream:
    """Builds the first message of the stream: AudioStreamHeader wrapped in AudioStream."""
    return AudioStream(
        audio_stream_header=AudioStreamHeader(
            audio_header=AudioHeader(
                samples_per_second=samplerate,
                bits_per_sample=bit_depth,
                channel_count=channels,
                audio_format=REQUIRED_AUDIO_FORMAT,
            ),
            emotion_post_processing_params=EmotionPostProcessingParameters(
                **config["post_processing_parameters"]
            ),
            face_params=FaceParameters(float_params=config["face_parameters"]),
            blendshape_params=BlendShapeParameters(
                bs_weight_multipliers=config["blendshape_parameters"]["multipliers"],
                bs_weight_offsets=config["blendshape_parameters"]["offsets"],
                enable_clamping_bs_weight=config["blendshape_parameters"]["enable_clamping_bs_weight"],
            ),
            emotion_params=EmotionParameters(
                live_transition_time=config["live_transition_time"],
                beginning_emotion=config["beginning_emotion"],
            ),
        )
    )


def _build_emotion_timecode_list(config: dict, preferred_emotion: dict[str, float] | None = None):
    """
    Builds the list of EmotionWithTimeCode.

    If preferred_emotion is given (a dict of A2F emotion name -> weight,
    e.g. {"joy": 1.0}, sourced from the classifier for this sentence), it
    replaces config['emotion_with_timecode_list'] entirely with a single
    timecode at t=0.

    IMPORTANT (Phase 8 fix): the sparse dict is expanded to a full 6-key
    vector before being sent. A2F's gRPC API expects all 6 emotion weights
    to be present; a partial dict was previously likely being silently
    ignored, which is why preferred_emotion had no visible effect on the
    avatar's face despite being wired through correctly.

    Falls back to the static YAML list when preferred_emotion is None
    (e.g. wav_sender.py's file-replay path, or a "neutral" chunk where
    bridge_server.py deliberately wants A2E's own audio-based inference
    to drive the face instead).
    """
    if preferred_emotion is not None:
        full_vector = _build_full_emotion_vector(preferred_emotion)
        logger.info(
            "A2F EMOTION VECTOR (full 6-key): %s  "
            "[sparse input was: %s]",
            full_vector, preferred_emotion,
        )
        return [EmotionWithTimeCode(emotion=full_vector, time_code=0.0)]

    return [
        EmotionWithTimeCode(emotion={**v["emotions"]}, time_code=v["time_code"])
        for v in config["emotion_with_timecode_list"].values()
    ]


async def send_audio(
    stream,
    samplerate: int,
    audio_data: np.ndarray,
    config: dict,
    preferred_emotion: dict[str, float] | None = None,
):
    """
    Sends audio + config over an open ProcessAudioStream() stream.

    Used for the WAV file path (wav_sender.py), where the full audio
    buffer is already loaded in memory before sending begins, and by the
    live bridge_server.py pipeline via receiver_streaming.py.

    Args:
        stream: the bidirectional stream returned by
                Audio2FaceClient.get_stream() (client.py)
        samplerate: sample rate of the audio, in Hz
        audio_data: 1-D numpy array of mono PCM16 samples
        config: parsed dict from the A2F emotion/face/blendshape YAML
                (face_parameters, blendshape_parameters,
                emotion_with_timecode_list, live_transition_time,
                beginning_emotion, post_processing_parameters)
        preferred_emotion: optional dict[str, float] of {a2f_emotion:
                weight}, one entry, overriding config's static
                emotion_with_timecode_list for this call only. None
                (default) preserves the old static-YAML behavior.
                NOTE: will be expanded to a full 6-key vector internally
                before being sent to A2F.
    """
    bit_depth = audio_data.dtype.itemsize * 8
    channels = 1 if audio_data.ndim == 1 else audio_data.shape[1]

    if bit_depth != REQUIRED_BITS_PER_SAMPLE:
        raise Audio2FaceStreamError("Unsupported audio type. Only 16-bit audio is supported.")
    if channels != REQUIRED_CHANNEL_COUNT:
        raise Audio2FaceStreamError("Unsupported audio type. Only mono audio is supported.")

    logger.info(f"Sending AudioStreamHeader ({samplerate} Hz, {bit_depth}-bit, mono)")
    await stream.write(_build_audio_stream_header(samplerate, bit_depth, channels, config))

    emotion_timecodes = _build_emotion_timecode_list(config, preferred_emotion)
    logger.debug("Emotion timecodes for this stream: %s", emotion_timecodes)

    chunk_count = len(audio_data) // samplerate + 1
    for i in range(chunk_count):
        chunk = audio_data[i * samplerate: i * samplerate + samplerate]
        if len(chunk) == 0:
            continue

        if i == 0:
            # Emotion timecodes are sent alongside the first audio chunk.
            await stream.write(
                AudioStream(
                    audio_with_emotion=AudioWithEmotion(
                        audio_buffer=chunk.astype(np.int16).tobytes(),
                        emotions=emotion_timecodes,
                    )
                )
            )
        else:
            await stream.write(
                AudioStream(
                    audio_with_emotion=AudioWithEmotion(
                        audio_buffer=chunk.astype(np.int16).tobytes()
                    )
                )
            )
        logger.debug(f"Sent audio chunk {i + 1}/{chunk_count}")

    logger.info("Sending EndOfAudio")
    await stream.write(AudioStream(end_of_audio=AudioStream.EndOfAudio()))


async def send_audio_stream(
    stream,
    samplerate: int,
    audio_chunks,
    config: dict,
    preferred_emotion: dict[str, float] | None = None,
):
    """
    Sends audio + config over an open ProcessAudioStream() stream, where
    audio arrives incrementally as an async generator/iterator of raw
    PCM16 mono byte chunks (e.g. from a streaming TTS engine), rather than
    a single pre-loaded numpy array.

    Args:
        stream: the bidirectional stream returned by
                Audio2FaceClient.get_stream() (client.py)
        samplerate: sample rate of the incoming audio, in Hz
        audio_chunks: an async generator/iterator yielding raw PCM16
                      mono bytes, in order, as they become available
        config: parsed dict from the A2F emotion/face/blendshape YAML
        preferred_emotion: optional dict[str, float] of {a2f_emotion:
                weight}, same semantics as send_audio()'s parameter --
                sent alongside the FIRST audio chunk, same as send_audio().
                NOTE: will be expanded to a full 6-key vector internally
                before being sent to A2F.
    """
    bit_depth = REQUIRED_BITS_PER_SAMPLE
    channels = REQUIRED_CHANNEL_COUNT

    logger.info(f"Sending AudioStreamHeader ({samplerate} Hz, {bit_depth}-bit, mono, streamed)")
    await stream.write(_build_audio_stream_header(samplerate, bit_depth, channels, config))

    emotion_timecodes = _build_emotion_timecode_list(config, preferred_emotion)
    logger.debug("Emotion timecodes for this stream: %s", emotion_timecodes)

    chunk_index = 0
    async for chunk in audio_chunks:
        if not chunk:
            continue

        if chunk_index == 0:
            # Emotion timecodes are sent alongside the first audio chunk,
            # same convention as send_audio() above.
            await stream.write(
                AudioStream(
                    audio_with_emotion=AudioWithEmotion(
                        audio_buffer=chunk,
                        emotions=emotion_timecodes,
                    )
                )
            )
        else:
            await stream.write(
                AudioStream(
                    audio_with_emotion=AudioWithEmotion(audio_buffer=chunk)
                )
            )
        chunk_index += 1
        logger.debug(f"Sent streamed audio chunk {chunk_index}")

    logger.info("Sending EndOfAudio")
    await stream.write(AudioStream(end_of_audio=AudioStream.EndOfAudio()))