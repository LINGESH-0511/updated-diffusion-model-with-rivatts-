import wave
import math
import struct
from dotenv import load_dotenv
load_dotenv()
from backend.rag.stt import transcribe_audio_bytes

# create a 1-second 440Hz sine wave as WAV
with wave.open("test.wav", "w") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    for i in range(16000):
        val = int(32767.0 * math.sin(2.0 * math.pi * 440.0 * i / 16000.0))
        w.writeframes(struct.pack('<h', val))

with open("test.wav", "rb") as f:
    data = f.read()

try:
    print(transcribe_audio_bytes(data, "test.wav"))
except Exception as e:
    print("Error:", e)
