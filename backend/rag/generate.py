"""
backend/rag/generate.py

Groq LLM generation, streaming. Plugs into the TTS/A2F chunking pipeline
in bridge_server.py.

UPDATED this session: system prompt now emits exactly ONE leading
[emotion] tag before the entire answer (same convention as websearch.py),
instead of a per-sentence tag on every sentence. bridge_server.py's
_handle_ask() collects the full streamed answer, parses the single leading
tag, and applies that one emotion to every TTS/A2F chunk so facial
expression and prosody stay consistent across the whole response.

Trade-off: we still stream tokens from Groq for efficiency, but
bridge_server.py now waits for the full answer before speaking, which
trades away the old "start speaking mid-generation" latency win. The gain
is consistent emotion across the answer instead of per-sentence mood swings.

Valid emotion tags the LLM can emit (maps through bridge_server.py's
_EMOTION_REMAP onto A2F's real 6-value facial set):
  [neutral]      -> A2E audio inference drives the face (neutral path)
  [friendly]     -> joy  (remapped)
  [caring]       -> neutral (remapped; TTS prosody carries the warmth)
  [professional] -> neutral (remapped)
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

from groq import Groq
from . import config
from .retrieve import hybrid_search, build_context_block

logger = logging.getLogger(__name__)

_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


# Single leading tag before the WHOLE answer — not per sentence.
# bridge_server.py's _handle_ask() parses this tag once and applies
# it to every chunk of the answer so facial expression is consistent.
SYSTEM_PROMPT = (
    "You are ARIA, a professional and empathetic AI avatar assistant. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain the answer, acknowledge this clearly "
    "and do not speculate or fabricate information. "
    "Responses must be concise, natural, and suitable for text-to-speech delivery.\n\n"
    "EMOTION TAGGING (mandatory):\n"
    "Begin every response with exactly ONE emotion tag on its own line. "
    "This tag controls the avatar's facial expression and voice tone. "
    "Select the most contextually appropriate tag using the rules below:\n\n"
    "  [supportive]   — Select when the user appears upset, distressed, "
    "frustrated, or is expressing negative emotions. Also select for any "
    "response that is empathetic, comforting, or reassuring in nature.\n\n"
    "  [sad]          — Select for any topic involving loss of life, death, "
    "natural disasters, accidents, tragedies, violence, war, or other "
    "somber and serious subject matter.\n\n"
    "  [friendly]     — Select for positive, uplifting, celebratory, or "
    "light-hearted topics.\n\n"
    "  [professional] — Select only for purely technical, scientific, or "
    "business topics that contain no emotional or tragic elements.\n\n"
    "Format: place the tag on its own line, followed immediately by your response.\n"
    "Example:\n"
    "[friendly]\n"
    "Based on the documents, here is what I found...\n\n"
    "One tag only. Do not repeat the tag within the response body."
)


def answer_query_stream(query: str, top_k: int | None = None) -> Iterator[str]:
    """
    Retrieve context, then stream the LLM's answer token-by-token as plain
    text chunks. The first tokens will be the leading [emotion] tag;
    bridge_server.py's _handle_ask() collects everything and parses the
    tag before starting TTS.
    """
    hits = hybrid_search(query, top_k=top_k)
    if not hits:
        yield "[neutral]\nI don't have any documents ingested yet, so I can't answer that."
        return

    context = build_context_block(hits)
    stream = _get_groq_client().chat.completions.create(
        model=config.GROQ_LLM_MODEL,
        temperature=config.GROQ_LLM_TEMPERATURE,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta


def transcribe_audio(audio_path: str) -> str:
    """
    Groq Whisper STT for the voice-query path.
    """
    client = _get_groq_client()
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=f,
            model=config.GROQ_STT_MODEL,
        )
    return result.text