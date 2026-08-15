import asyncio
import numpy as np
from backend.audio2face.receiver_streaming import stream_animation_frames
from backend.audio2face.wav_sender import load_config
import traceback

async def test():
    try:
        config = load_config('Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config_james.yml')
        async for f in stream_animation_frames(b'\x00'*1000, 22050, config):
            print(f)
    except Exception as e:
        traceback.print_exc()

asyncio.run(test())
