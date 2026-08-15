import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from backend.rag.stt import transcribe_audio_bytes

with open("test.webm", "wb") as f:
    f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

with open("test.webm", "rb") as f:
    data = f.read()

try:
    print("Sending audio to STT...")
    print(transcribe_audio_bytes(data))
except Exception as e:
    print("Error:", e)
