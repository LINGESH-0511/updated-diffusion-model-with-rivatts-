"""
backend/rag/stt.py - Groq Whisper speech-to-text for voice input.

Separate from generate.py (Groq LLM calls) since this hits a different
Groq endpoint (audio/transcriptions, multipart file upload, non-streaming)
with its own error shape -- keeping it isolated avoids tangling the RAG
generation streaming code with a plain synchronous HTTP call.
"""
import requests

from backend.rag import config
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Whisper's own input limit; the WS-level check in bridge_server.py should
# stay at or below this so a rejected clip fails fast instead of round-
# tripping to Groq first.
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def transcribe_audio_bytes(audio_bytes: bytes, filename: str = "voice_input.webm") -> str:
    """
    Sends raw audio bytes to Groq's hosted Whisper endpoint and returns the
    transcript text. Whatever MediaRecorder produces in the browser
    (webm/opus) works fine -- no client-side conversion needed.

    Raises RuntimeError with Groq's own error message on failure, so the
    caller can surface something useful over the WebSocket instead of a
    generic traceback.
    """
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set -- cannot transcribe audio.")

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise RuntimeError(
            f"Audio too large ({len(audio_bytes)} bytes, max {MAX_AUDIO_BYTES})"
        )

    resp = requests.post(
        _GROQ_TRANSCRIBE_URL,
        headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
        files={"file": (filename, audio_bytes)},
        data={"model": config.GROQ_STT_MODEL},
        timeout=30,
    )

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text)
            if "too short" in detail.lower():
                raise RuntimeError("Audio was too short. Please hold the mic button down while speaking, then release.")
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            detail = resp.text
        raise RuntimeError(f"Groq transcription failed ({resp.status_code}): {detail}")

    return resp.json().get("text", "").strip()