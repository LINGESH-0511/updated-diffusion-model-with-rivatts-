"""
backend/tts/piper_streamer.py

SINGLE-VOICE version (replaces the 4-voice bucket system).

Why this changed: switching between 4 different Piper checkpoints
(Amy/Lessac/Kristin/Danny) mid-response made the avatar sound like a
different person every line -- audibly "unreal." One consistent voice
identity, with emotion expressed through prosody (pace/pitch-variance/
phoneme-width) instead of a different speaker, sounds far more natural
and matches how real speech actually works: your voice doesn't change
identity when your mood shifts, its *delivery* does.

Drop-in contract (unchanged):
    synthesize_pcm(text: str, emotion: str = "neutral") -> (pcm_bytes, sample_rate)

Emotion now maps to a PROSODY PRESET (length_scale/noise_scale/noise_w),
not a voice file. All 13 emotion values used elsewhere in the pipeline
(bridge_server.py's VALID_EMOTIONS: the 4 LLM-facing tags plus A2F's 10
facial emotions used as classifier fallback) resolve to one of these
presets. Add/tune entries in _EMOTION_PROSODY as needed -- this is the
only place emotion->sound behavior is decided now.

piper1-gpl 1.4.x API note:
    synthesize() returns Iterable[AudioChunk]; each chunk has an
    .audio_int16_bytes attribute containing raw int16 PCM bytes.
    synthesize() accepts a SynthesisConfig for per-call length_scale/
    noise_scale/noise_w overrides -- see _run_inference().
"""

import os
import threading
from typing import Tuple

from backend.utils.logger import get_logger

logger = get_logger(__name__)

_USE_CUDA = os.environ.get("PIPER_USE_CUDA", "0") == "1"

# The ONE voice this project uses now. Swap the env var / default if you
# want a different single checkpoint -- everything downstream is voice-
# agnostic as long as this one path is valid.
_VOICE_PATH = os.environ.get(
    "PIPER_VOICE",
    os.path.join(os.path.dirname(__file__), "..", "..", "voices", "neutral", "en_US-amy-medium.onnx"),
)

# Piper's per-call synthesis knobs, tuned per emotion instead of per voice.
#   length_scale: >1.0 = slower/more deliberate, <1.0 = faster/brisker
#   noise_scale:  overall vocal "texture"/variance -- higher = more
#                 expressive/breathy, lower = flatter/steadier
#   noise_w:      phoneme-duration variability -- higher = more natural
#                 rhythmic variation, lower = more even/robotic
# Defaults below are a reasonable starting point -- ears are the real
# tuning tool, adjust to taste once you can hear it live.
_DEFAULT_PROSODY = {"length_scale": 1.0, "noise_scale": 0.667, "noise_w": 0.8}

_EMOTION_PROSODY = {
    "neutral":      {"length_scale": 1.00, "noise_scale": 0.667, "noise_w": 0.80},
    "professional": {"length_scale": 0.92, "noise_scale": 0.500, "noise_w": 0.65},
    "friendly":     {"length_scale": 0.97, "noise_scale": 0.750, "noise_w": 0.90},
    "supportive":   {"length_scale": 1.12, "noise_scale": 0.700, "noise_w": 0.85},
    "joy":          {"length_scale": 0.93, "noise_scale": 0.800, "noise_w": 0.95},
    "amazement":    {"length_scale": 0.95, "noise_scale": 0.800, "noise_w": 0.95},
    "cheekiness":   {"length_scale": 0.95, "noise_scale": 0.780, "noise_w": 0.92},
    "sad":          {"length_scale": 1.20, "noise_scale": 0.550, "noise_w": 0.60},
    "sadness":      {"length_scale": 1.20, "noise_scale": 0.550, "noise_w": 0.60},
    "grief":        {"length_scale": 1.25, "noise_scale": 0.500, "noise_w": 0.55},
    "fear":         {"length_scale": 1.05, "noise_scale": 0.780, "noise_w": 0.95},
    "pain":         {"length_scale": 1.10, "noise_scale": 0.700, "noise_w": 0.90},
    "disgust":      {"length_scale": 1.05, "noise_scale": 0.600, "noise_w": 0.70},
    "anger":        {"length_scale": 0.90, "noise_scale": 0.850, "noise_w": 1.00},
    "outofbreath":  {"length_scale": 0.90, "noise_scale": 0.800, "noise_w": 0.95},
}
_DEFAULT_EMOTION = "neutral"

_model_lock = threading.Lock()
_voice = None  # single loaded PiperVoice, lazy-loaded once


def _load_voice():
    model_path = _VOICE_PATH
    config_path = model_path + ".json"

    if not os.path.isfile(model_path):
        raise RuntimeError(f"Piper model not found: {model_path!r}")
    if not os.path.isfile(config_path):
        raise RuntimeError(f"Piper config not found: {config_path!r}")

    try:
        from piper import PiperVoice
    except ImportError as e:
        raise RuntimeError(
            "Cannot import 'piper.PiperVoice'. Run: pip install piper-tts"
        ) from e

    voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=_USE_CUDA)
    logger.info(
        "Loaded Piper voice: %s (sample_rate=%s)",
        model_path, getattr(getattr(voice, "config", None), "sample_rate", "?"),
    )
    return voice


def _get_voice():
    global _voice
    if _voice is None:
        with _model_lock:
            if _voice is None:
                _voice = _load_voice()
    return _voice


def preload_voice():
    """Call once at server startup so the first request doesn't pay model-load latency."""
    _get_voice()


def _resolve_prosody(emotion: str) -> dict:
    emotion = (emotion or _DEFAULT_EMOTION).strip().lower()
    prosody = _EMOTION_PROSODY.get(emotion)
    if prosody is None:
        logger.warning("Unmapped emotion %r, falling back to '%s' prosody", emotion, _DEFAULT_EMOTION)
        return _EMOTION_PROSODY[_DEFAULT_EMOTION]
    return prosody


def _run_inference(voice, text: str, prosody: dict) -> Tuple[bytes, int]:
    sample_rate = getattr(voice.config, "sample_rate", 22050)

    # Path (a): older rhasspy piper-tts API -- no per-call prosody override
    # support in this code path, so prosody is applied only when using
    # path (b) below. If your installed piper-tts only exposes
    # synthesize_stream_raw, upgrade piper-tts to get prosody control.
    if hasattr(voice, "synthesize_stream_raw") and not hasattr(voice, "synthesize"):
        logger.warning(
            "Installed piper-tts only exposes synthesize_stream_raw() -- "
            "per-emotion prosody will NOT be applied. Upgrade piper-tts "
            "(pip install -U piper-tts) to enable prosody control."
        )
        pcm_bytes = b"".join(voice.synthesize_stream_raw(text))
        return pcm_bytes, sample_rate

    # Path (b): piper1-gpl 1.4.x -- synthesize() accepts a syn_config
    # kwarg (SynthesisConfig) for per-call length_scale/noise_scale/noise_w.
    if hasattr(voice, "synthesize"):
        try:
            from piper import SynthesisConfig
            syn_config = SynthesisConfig(
                length_scale=prosody["length_scale"],
                noise_scale=prosody["noise_scale"],
                noise_w=prosody["noise_w"],
            )
            chunks = list(voice.synthesize(text, syn_config=syn_config))
        except (ImportError, TypeError):
            # SynthesisConfig not available in this piper-tts version --
            # fall back to default prosody rather than crashing the turn.
            logger.warning(
                "SynthesisConfig unavailable in installed piper-tts -- "
                "synthesizing with default prosody (emotion=%s ignored). "
                "Upgrade piper-tts for prosody control.",
                prosody,
            )
            chunks = list(voice.synthesize(text))
        pcm_bytes = b"".join(chunk.audio_int16_bytes for chunk in chunks)
        return pcm_bytes, sample_rate

    raise RuntimeError(
        "Installed piper-tts version exposes neither synthesize_stream_raw() "
        "nor synthesize() on PiperVoice -- check pip show piper-tts."
    )


def synthesize_pcm(text: str, emotion: str = _DEFAULT_EMOTION) -> Tuple[bytes, int]:
    """
    Synchronous: (text, emotion) -> (raw_pcm_bytes, sample_rate).
    Always uses the single loaded voice; `emotion` only selects the
    prosody preset applied to that one voice.
    """
    prosody = _resolve_prosody(emotion)
    voice = _get_voice()
    pcm_bytes, sample_rate = _run_inference(voice, text, prosody)
    logger.debug(
        "synthesize_pcm (Piper): %d chars -> %d bytes @ %d Hz (emotion=%s -> prosody=%s)",
        len(text), len(pcm_bytes), sample_rate, emotion, prosody,
    )
    return pcm_bytes, sample_rate