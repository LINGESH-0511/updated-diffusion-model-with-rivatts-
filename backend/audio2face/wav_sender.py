"""
backend/audio2face/wav_sender.py

Phase 4: Read a WAV file + A2F config YAML from disk and send them to
Audio2Face via stream.py's send_audio().

Responsibilities:
    - Read WAV file (sample rate + PCM16 mono data)
    - Load the A2F emotion/face/blendshape config YAML
    - Hand both off to stream.send_audio() to chunk + stream

Does NOT:
    - Open the gRPC channel (client.py's job)
    - Build/send protobuf messages directly (stream.py's job)
    - Read the response / blendshapes back (receiver.py's job)
"""

import warnings

import scipy.io.wavfile
import yaml

from backend.audio2face.stream import send_audio
from backend.utils.logger import get_logger
from backend.utils.exceptions import Audio2FaceStreamError

logger = get_logger(__name__)

REQUIRED_BITS_PER_SAMPLE = 16
REQUIRED_CHANNEL_COUNT = 1


def load_config(config_path: str) -> dict:
    """Loads and parses the A2F config YAML (face/blendshape/emotion parameters)."""
    logger.info(f"Loading Audio2Face config from {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_wav(wav_path: str):
    """
    Reads a WAV file and validates it against Audio2Face requirements
    (16-bit PCM, mono).

    Returns:
        (samplerate, data) tuple, ready to pass into stream.send_audio()
    """
    warnings.filterwarnings("ignore", category=scipy.io.wavfile.WavFileWarning)

    logger.info(f"Reading WAV file from {wav_path}")
    samplerate, data = scipy.io.wavfile.read(wav_path)

    bit_depth = data.dtype.itemsize * 8
    channels = data.shape[1] if data.ndim > 1 else 1

    logger.info(f"WAV info: {samplerate} Hz, {bit_depth}-bit, {channels} channel(s)")

    if bit_depth != REQUIRED_BITS_PER_SAMPLE:
        raise Audio2FaceStreamError(
            f"Unsupported WAV file '{wav_path}': only 16-bit audio is supported, got {bit_depth}-bit."
        )
    if channels != REQUIRED_CHANNEL_COUNT:
        raise Audio2FaceStreamError(
            f"Unsupported WAV file '{wav_path}': only mono audio is supported, got {channels} channels."
        )

    return samplerate, data


async def send_wav_file(stream, wav_path: str, config_path: str):
    """
    Reads a WAV file + config from disk and streams it to Audio2Face
    over an already-open ProcessAudioStream() stream.

    Args:
        stream: the bidirectional stream returned by
                Audio2FaceClient.get_stream() (client.py)
        wav_path: path to a 16-bit PCM mono WAV file
        config_path: path to the A2F face/blendshape/emotion config YAML
    """
    config = load_config(config_path)
    samplerate, data = load_wav(wav_path)

    logger.info(f"Streaming '{wav_path}' to Audio2Face")
    await send_audio(stream, samplerate, data, config)
    logger.info("Finished streaming WAV file")