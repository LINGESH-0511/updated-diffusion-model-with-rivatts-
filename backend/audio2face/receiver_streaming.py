"""
backend/audio2face/receiver_streaming.py

Streaming variant of receiver.py — yields one blendshape frame dict
the instant it arrives from the A2F gRPC stream, instead of collecting
everything into AnimationResult first.

Mirrors receiver.py's field access exactly:
  - message.animation_data_stream_header  → bs_names
  - message.animation_data.skel_animation.blend_shape_weights → per-frame values
  - message.status                        → final status

Yielded dicts:
  {"type": "blendshapes", "frameIndex": N, "blendShapes": {"JawOpen": 0.3, ...}, "timeCode": T}
  {"type": "end", "totalFrames": N, "statusCode": C, "statusMessage": "..."}

Usage (from bridge_server.py):
    async for frame in stream_animation_frames(pcm_bytes, sample_rate, config):
        await websocket.send_json(frame)

Emotion wiring (this revision): stream_animation_frames() now accepts an
optional `preferred_emotion` dict, forwarded straight through to
stream.send_audio() -- see stream.py's _build_emotion_timecode_list() for
how it overrides the static YAML emotion_with_timecode_list. None (default)
preserves the original static behavior.
"""

import asyncio

import numpy as np
import grpc

from nvidia_ace.animation_data.v1_pb2 import AnimationData, AnimationDataStreamHeader

from backend.audio2face.client import Audio2FaceClient
from backend.audio2face.stream import send_audio
from backend.utils.logger import get_logger

logger = get_logger(__name__)


async def stream_animation_frames(
    pcm_bytes: bytes,
    sample_rate: int,
    config: dict,
    preferred_emotion: dict[str, float] | None = None,
):
    """
    Async generator.

    Opens Audio2FaceClient, sends the full PCM audio via send_audio(),
    then yields each blendshape frame as a dict the moment it arrives.

    Args:
        pcm_bytes:   Raw 16-bit mono PCM bytes (from synthesize_pcm()).
        sample_rate: Sample rate of the audio (e.g. 24000 for Chatterbox-Turbo).
        config:      Parsed A2F YAML config dict (from load_config()).
        preferred_emotion: optional dict[str, float] of {a2f_emotion: weight},
                     e.g. {"joy": 1.0} -- forwarded to send_audio() to override
                     the static emotion_with_timecode_list for this call only.
                     None lets A2E's own audio-based inference drive facial
                     emotion instead.

    Yields:
        {"type": "blendshapes", "frameIndex": int, "blendShapes": dict, "timeCode": float}
        {"type": "end", "totalFrames": int, "statusCode": int, "statusMessage": str}
    """
    # Convert raw PCM bytes → int16 numpy array (what send_audio() expects)
    audio_data = np.frombuffer(pcm_bytes, dtype=np.int16)

    async with Audio2FaceClient() as client:
        stream = client.get_stream()

        # Send audio + config to A2F (non-blocking — gRPC streams writes async)
        send_task = asyncio.create_task(
            send_audio(stream, sample_rate, audio_data, config, preferred_emotion=preferred_emotion)
        )

        bs_names = []
        frame_index = 0
        status_code = None
        status_message = ""

        # Read response frames as they arrive
        while True:
            message = await stream.read()

            if message == grpc.aio.EOF:
                logger.info("A2F stream EOF — %d frames total", frame_index)
                break

            if message.HasField("animation_data_stream_header"):
                header: AnimationDataStreamHeader = message.animation_data_stream_header
                bs_names = list(header.skel_animation_header.blend_shapes)
                logger.info("A2F header received — %d blendshapes: %s…",
                            len(bs_names), bs_names[:5])

            elif message.HasField("animation_data"):
                animation_data: AnimationData = message.animation_data

                for bsw in animation_data.skel_animation.blend_shape_weights:
                    bs_dict = dict(zip(bs_names, bsw.values))
                    logger.debug("Frame %d JawOpen=%.3f", frame_index, bs_dict.get("JawOpen", 0))
                    yield {
                        "type": "blendshapes",
                        "frameIndex": frame_index,
                        "blendShapes": bs_dict,
                        "timeCode": float(bsw.time_code),
                    }
                    frame_index += 1

            elif message.HasField("status"):
                status = message.status
                status_code = status.code
                status_message = status.message
                names = {0: "SUCCESS", 1: "INFO", 2: "WARNING", 3: "ERROR"}
                logger.info("A2F status: %s — %s",
                            names.get(status_code, status_code), status_message)

        # Make sure send_audio() finished cleanly
        try:
            await send_task
        except Exception as e:
            logger.warning("send_audio task error (may be harmless): %s", e)

    yield {
        "type": "end",
        "totalFrames": frame_index,
        "statusCode": status_code,
        "statusMessage": status_message,
    }