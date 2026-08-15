"""
backend/audio2face/save_animation.py

Runs the same WAV -> Audio2Face pipeline as test_audio.py, but saves
the resulting blendshape animation + audio to disk as JSON + WAV,
so the frontend avatar renderer can be built and tested without
Docker / A2F needing to be running every time.

Run with (from D:\\A2F):
    python -m backend.audio2face.save_animation --wav "Audio2Face-3D-Samples/example_audio/Mark_neutral.wav" --config "Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config_james.yml" --out "frontend/assets/test_animation.json"
"""

import argparse
import asyncio
import json
import wave

from backend.audio2face.client import Audio2FaceClient
from backend.audio2face.wav_sender import send_wav_file
from backend.audio2face.receiver import read_animation_stream
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Save an A2F animation result to disk for renderer testing")
    parser.add_argument("--wav", required=True, help="Path to a 16-bit PCM mono WAV file")
    parser.add_argument("--config", required=True, help="Path to the A2F config YAML")
    parser.add_argument("--out", default="frontend/assets/test_animation.json", help="Output JSON path")
    parser.add_argument("--out-wav", default="frontend/assets/test_animation_audio.wav", help="Output WAV path (audio echoed back by A2F)")
    return parser.parse_args()


async def run(wav_path: str, config_path: str, out_json: str, out_wav: str):
    async with Audio2FaceClient() as client:
        stream = client.get_stream()
        send_task = asyncio.create_task(send_wav_file(stream, wav_path, config_path))
        receive_task = asyncio.create_task(read_animation_stream(stream))
        await send_task
        result = await receive_task

    if not result.animation_key_frames:
        logger.error("No animation frames received - nothing to save. Is Docker running?")
        return

    data = {
        "bs_names": result.bs_names,
        "frames": result.animation_key_frames,  # [{timeCode, blendShapes: {...}}, ...]
        "status_code": result.status_code,
        "status_message": result.status_message,
    }
    with open(out_json, "w") as f:
        json.dump(data, f)
    logger.info(f"Saved {len(result.animation_key_frames)} frames -> {out_json}")

    if result.audio_buffer and result.audio_header:
        with wave.open(out_wav, "wb") as wf:
            wf.setnchannels(result.audio_header.channel_count or 1)
            wf.setsampwidth((result.audio_header.bits_per_sample or 16) // 8)
            wf.setframerate(result.audio_header.samples_per_second or 22050)
            wf.writeframes(result.audio_buffer)
        logger.info(f"Saved audio -> {out_wav}")


def main():
    args = parse_args()
    asyncio.run(run(args.wav, args.config, args.out, args.out_wav))


if __name__ == "__main__":
    main()
