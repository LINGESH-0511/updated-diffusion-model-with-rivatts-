"""
backend/emotion/classifier.py

Sentence-level emotion classification using a small DistilRoBERTa model,
mapped onto A2F's own emotion vocabulary (see bridge_server.py's
VALID_EMOTIONS) so downstream code needs zero changes.

CHANGED THIS REVISION:
  - Added classify_emotion_for_response(), which should be called ONCE
    on the full response text (not per sentence) when the LLM path
    (RAG/websearch) fails to supply its own leading emotion tag. This
    keeps the classifier fallback consistent with the "one emotion per
    turn" decision made for TTS voice/prosody and A2F facial emotion --
    the fallback shouldn't reintroduce per-sentence flapping that the
    LLM-tag path was specifically changed to avoid.
  - classify_emotion() (the original per-text function) is unchanged and
    still available if you ever want per-sentence classification for
    something else -- classify_emotion_for_response() is just a thin
    wrapper that makes the "call it once, on the whole text" contract
    explicit and hard to misuse.

PREVIOUSLY FIXED:
  - device=0 (GPU) was crashing silently every call whenever it collided
    with A2F/CosyVoice3 for VRAM -- classify_emotion()'s broad
    `except Exception` swallowed the error and returned "neutral" every
    time, with zero visible symptom other than "everything is neutral".
    _get_classifier() tries GPU first, falls back to CPU automatically
    if GPU load fails, and logs which device it actually ended up on.
"""
from transformers import pipeline as hf_pipeline
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
_CONFIDENCE_THRESHOLD = 0.4

_classifier = None


def _get_classifier():
    global _classifier
    if _classifier is None:
        try:
            logger.info("Loading emotion classifier on GPU (device=0): %s", _MODEL_NAME)
            _classifier = hf_pipeline(
                "text-classification",
                model=_MODEL_NAME,
                top_k=1,
                device=0,
            )
            logger.info("Emotion classifier loaded successfully on GPU")
        except Exception:
            logger.exception(
                "Emotion classifier failed to load on GPU -- falling back to CPU. "
                "This is likely VRAM contention with A2F/CosyVoice3."
            )
            _classifier = hf_pipeline(
                "text-classification",
                model=_MODEL_NAME,
                top_k=1,
                device=-1,
            )
            logger.info("Emotion classifier loaded successfully on CPU")
    return _classifier


# Maps the model's 7 labels onto the 6 emotions A2F's preferred_emotion
# actually accepts: anger, disgust, fear, joy, neutral, sadness.
# NOTE: the classifier's native "surprise" label has no direct match in
# that 6-value set (there is no "amazement"/"surprise" slot on A2F here).
# Mapped to "joy" as the closest fit (surprise is usually positive-valence
# in casual speech) -- change to "neutral" instead if that reads better
# on the avatar's face once you can see it live.
_LABEL_MAP = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "neutral",
    "fear": "sadness",
    "disgust": "neutral",
    "surprise": "joy",  # closest fit in the 6-emotion A2F set -- tune if needed
    "neutral": "neutral",
}


def classify_emotion(text: str) -> str:
    """
    Returns one of bridge_server.py's VALID_EMOTIONS for the given text,
    or 'neutral' if confidence is too low or text is empty.

    NOTE: this can be called on any span of text (a sentence, a
    paragraph, a full response). For the "one emotion per turn" fallback
    path, use classify_emotion_for_response() below instead of calling
    this directly per sentence -- see module docstring.
    """
    text = (text or "").strip()
    if not text:
        return "neutral"

    try:
        result = _get_classifier()(text)[0][0]
    except Exception:
        logger.exception("Emotion classification failed, falling back to neutral")
        return "neutral"

    raw_label = result["label"]
    score = result["score"]
    mapped_label = _LABEL_MAP.get(raw_label, "neutral")

    if score < _CONFIDENCE_THRESHOLD:
        logger.info(
            "CLASSIFIER: %r -> raw=%s (score=%.3f, BELOW threshold %.2f) -> forcing neutral",
            text[:80], raw_label, score, _CONFIDENCE_THRESHOLD,
        )
        return "neutral"

    logger.info(
        "CLASSIFIER: %r -> raw=%s (score=%.3f) -> mapped=%s",
        text[:80], raw_label, score, mapped_label,
    )
    return mapped_label


def classify_emotion_for_response(full_response_text: str) -> str:
    """
    Fallback path for when the LLM didn't supply its own leading emotion
    tag. Call this ONCE with the entire response text (all sentences
    joined), not per sentence -- classifying per sentence here would
    reintroduce the "different mood every line" problem that the LLM-tag
    path was changed to avoid (see websearch.py's _SYSTEM_PROMPT and
    piper_streamer.py's single-voice/prosody rewrite).

    If you want a slightly better signal than "classify the whole blob
    at once" without going back to per-sentence, a reasonable middle
    ground is classifying just the first 1-2 sentences (usually where a
    response's overall tone is set) instead of the full text -- but
    default to the full text unless you've specifically tuned this.
    """
    return classify_emotion(full_response_text)


def warm_up_classifier():
    """Call this from bridge_server.py's warm_up() to pay load cost upfront."""
    classify_emotion("This is a warm up sentence.")