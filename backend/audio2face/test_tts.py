"""
backend/audio2face/test_tts.py

Phase 6: End-to-end verification.

    Text  -->  Piper TTS (local, streaming)  -->  Audio2Face  -->  Avatar

Run with:
    python -m backend.audio2face.test_tts --text "Hello, this is a test." --model voices/en_US-lessac-medium.onnx
"""

import argparse
import asyncio
import time

from backend.audio2face.client import Audio2FaceClient
from backend.audio2face.stream import send_audio_stream
from backend.audio2face.receiver import read_animation_stream
from backend.audio2face.wav_sender import load_config
from backend.tts.piper_streamer import stream_tts_audio, get_sample_rate
from backend.utils.logger import get_logger
from backend.utils.exceptions import Audio2FaceConnectionError, Audio2FaceStreamError

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = "Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config_mark.yml"
DEFAULT_MODEL_PATH = "voices/en_US-lessac-medium.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Test the Piper TTS -> Audio2Face -> Avatar pipeline")
    parser.add_argument("--text", required=True, help="Text to synthesize and send to Audio2Face")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the Piper .onnx voice model")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to the A2F config YAML")
    return parser.parse_args()


async def run_test(text: str, model_path: str, config_path: str):
    logger.info("=== Audio2Face Piper TTS streaming test ===")
    logger.info(f"Text:  {text}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Config file: {config_path}")

    config = load_config(config_path)
    samplerate = get_sample_rate(model_path)
    start_time = time.time()

    async with Audio2FaceClient() as client:
        stream = client.get_stream()

        audio_chunks = stream_tts_audio(text, model_path)

        send_task = asyncio.create_task(
            send_audio_stream(stream, samplerate, audio_chunks, config)
        )
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
        print("\nSUCCESS: Text -> Piper TTS -> Audio2Face -> Talking Avatar pipeline is working.")
    else:
        print("\nFAILED: No animation data received, or status was not SUCCESS.")

    return result


def main():
    args = parse_args()
    try:
        asyncio.run(run_test(args.text, args.model, args.config))
    except Audio2FaceConnectionError as e:
        logger.error(f"Connection error: {e}")
    except Audio2FaceStreamError as e:
        logger.error(f"Stream error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()