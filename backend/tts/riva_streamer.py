"""
backend/tts/riva_streamer.py

NVIDIA Riva / Magpie TTS via the NVIDIA API Catalog (REST).

Drop-in replacement for piper_streamer.py:
    synthesize_pcm(text: str, emotion: str = "neutral") -> (pcm_bytes, sample_rate)

Why REST instead of gRPC:
    The cloud-hosted Riva endpoint (grpc.nvcf.nvidia.com:443) requires a
    per-model "function-id" that is account-scoped and may not be available
    on all NGC plans. The REST API at integrate.api.nvidia.com uses the same
    NVIDIA_API_KEY without a per-function approval step, making it work for
    all accounts with API access. For local Riva deployment, swap the URL and
    remove the auth header.

Model: nvidia/magpie-tts-multilingual
    - Highest-quality English TTS on the API Catalog.
    - Supports multiple voices; we use "English-US.Female-1" by default.
    - Returns raw WAV bytes (PCM audio with header).

Emotion → speed/rate mapping:
    Riva's "rate" parameter accepts a float where 1.0 is normal speed.
    We approximate Piper's length_scale logic: emotions that Piper slowed
    down (sadness, grief) get a rate < 1.0, and fast emotions (anger, joy)
    get a rate > 1.0. The effect is subtle but consistent.
"""
from __future__ import annotations

import io
import os
import struct
import threading
import wave
from typing import Tuple

import riva.client
import requests

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ---- Config ----------------------------------------------------------------
_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
_API_URL = "https://integrate.api.nvidia.com/v1/audio/speech"
_MODEL   = os.environ.get("RIVA_TTS_MODEL", "nvidia/magpie-tts-multilingual")
_VOICE   = os.environ.get("RIVA_TTS_VOICE", "Magpie-Multilingual.EN-US.Aria")
_SAMPLE_RATE = 22050   # Riva returns 22050 Hz PCM for Magpie

# Emotion → speed rate (1.0 = normal).
_EMOTION_RATE: dict[str, float] = {
    "professional": 1.05,   # Slightly brisk, confident and efficient
    "friendly":     1.02,   # Natural, conversational upbeat pace
    "supportive":   0.92,   # Calmer, slower, comforting without sounding distorted
    "sad":          0.85,   # Noticeably slower, hesitant and somber
}

_DEFAULT_EMOTION = "professional"

# Riva gRPC call is thread-safe.
_service_lock = threading.Lock()
_service: riva.client.SpeechSynthesisService | None = None

def _get_service() -> riva.client.SpeechSynthesisService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                auth = riva.client.Auth(
                    uri="grpc.nvcf.nvidia.com:443",
                    use_ssl=True,
                    metadata_args=[
                        ("function-id", "877104f7-e885-42b9-8de8-f6e4c6303969"),
                        ("authorization", f"Bearer {_NVIDIA_API_KEY}"),
                    ],
                )
                _service = riva.client.SpeechSynthesisService(auth)
    return _service

def _resolve_rate(emotion: str) -> float:
    emotion = (emotion or _DEFAULT_EMOTION).strip().lower()
    return _EMOTION_RATE.get(emotion, _EMOTION_RATE[_DEFAULT_EMOTION])

def preload_voice() -> None:
    """
    'Warm up' the Riva TTS connection by making a tiny test call at startup.
    Validates the API key and surfaces auth errors early.
    """
    if not _NVIDIA_API_KEY:
        logger.warning("NVIDIA_API_KEY not set – Riva TTS will fail at call time.")
        return
    try:
        pcm, sr = synthesize_pcm("Hello.", "neutral")
        logger.info(
            "Riva TTS warm-up: synthesized %d bytes @ %d Hz (voice=%s)",
            len(pcm), sr, _VOICE,
        )
    except Exception:
        logger.exception("Riva TTS warm-up failed (will retry on first real request)")

def synthesize_pcm(text: str, emotion: str = _DEFAULT_EMOTION) -> Tuple[bytes, int]:
    """
    Synchronous: (text, emotion) -> (raw_pcm_int16_bytes, sample_rate).
    """
    if not _NVIDIA_API_KEY:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Cannot use Riva TTS. "
            "Set it in .env or environment."
        )

    rate = _resolve_rate(emotion)
    service = _get_service()
    
    # We pass rate via custom_configuration. Riva handles 'speed' or 'rate' depending on the model.
    # The SDK method is synthesize(text, voice_name, language_code, encoding, sample_rate_hz, custom_configuration=...)
    try:
        resp = service.synthesize(
            text=text,
            voice_name=_VOICE,
            language_code="en-US",
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hz=_SAMPLE_RATE,
            custom_configuration={"rate": str(rate)}
        )
    except Exception as e:
        raise RuntimeError(f"Riva TTS gRPC error: {e}") from e

    pcm_bytes = resp.audio
    
    logger.debug(
        "synthesize_pcm (Riva): %d chars -> %d bytes @ %d Hz "
        "(emotion=%s, rate=%.2f, voice=%s)",
        len(text), len(pcm_bytes), _SAMPLE_RATE, emotion, rate, _VOICE,
    )
    return pcm_bytes, _SAMPLE_RATE
