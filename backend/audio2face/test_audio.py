"""
backend/audio2face/test_audio.py

Phase 5: End-to-end verification.

    WAV  -->  Audio2Face  -->  Talking Avatar (blendshapes)

Wires together:
    - client.py     (open gRPC channel + stub)
    - wav_sender.py (read WAV + config, send audio)
    - receiver.py   (read blendshapes / emotion / status back)

Run with:
    python -m backend.audio2face.test_audio
or:
    python -m backend.audio2face.test_audio --wav path/to/file.wav --config path/to/config.yaml
"""

import argparse
import asyncio
import time

from backend.audio2face.client import Audio2FaceClient
from backend.audio2face.wav_sender import send_wav_file
from backend.audio2face.receiver import read_animation_stream
from backend.utils.logger import get_logger
from backend.utils.exceptions import Audio2FaceConnectionError, Audio2FaceStreamError

logger = get_logger(__name__)

# NOTE: adjust these two defaults to match your actual project folder layout.
DEFAULT_WAV_PATH = "assets/test.wav"
DEFAULT_CONFIG_PATH = "Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Test the WAV -> Audio2Face -> Avatar pipeline")
    parser.add_argument("--wav", default=DEFAULT_WAV_PATH, help="Path to a 16-bit PCM mono WAV file")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to the A2F config YAML")
    return parser.parse_args()


async def run_test(wav_path: str, config_path: str):
    logger.info("=== Audio2Face end-to-end test ===")
    logger.info(f"WAV file:    {wav_path}")
    logger.info(f"Config file: {config_path}")

    start_time = time.time()

    async with Audio2FaceClient() as client:
        stream = client.get_stream()

        # Sending and receiving must run concurrently: Audio2Face streams
        # blendshapes back while it is still receiving audio.
        send_task = asyncio.create_task(send_wav_file(stream, wav_path, config_path))
        receive_task = asyncio.create_task(read_animation_stream(stream))

        await send_task
        result = await receive_task

    elapsed = time.time() - start_time

    print("\n=== RESULT ===")
    print(f"Status code:    {result.status_code}  ({result.status_message})")
    print(f"Blendshapes:    {len(result.bs_names)} channels")
    print(f"Animation frames received: {len(result.animation_key_frames)}")
    print(f"Audio bytes received:      {len(result.audio_buffer)}")
    print(f"Total time: {elapsed:.2f}s")

    if result.status_code == 0 and result.animation_key_frames:
        print("\nSUCCESS: WAV -> Audio2Face -> Talking Avatar pipeline is working.")
    else:
        print("\nFAILED: No animation data received, or status was not SUCCESS.")

    return result


def main():
    args = parse_args()
    try:
        asyncio.run(run_test(args.wav, args.config))
    except Audio2FaceConnectionError as e:
        logger.error(f"Connection error: {e}")
    except Audio2FaceStreamError as e:
        logger.error(f"Stream error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()