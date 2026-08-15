"""
backend/websearch.py - Web-search answering via Groq's built-in "compound"
systems (server-side web search powered by Tavily, no separate search
account needed -- same GROQ_API_KEY that powers the RAG generation step).

Public contract (relied on by backend/bridge_server.py's _handle_web_ask):

    web_search_answer(question: str) -> {
        "answer":  str,
        "sources": [{"url": str, "title": str}, ...],
        "emotion": str,   # NEW -- one of VALID_EMOTIONS' A2F-facing set
    }

CHANGES THIS SESSION:
  1. BUG FIX: this module never returned an "emotion" key, so
     bridge_server.py's `result.get("emotion") or "neutral"` was silently
     always falling back to "neutral" -- web-mode facial emotion has been
     dead since it was wired in. Fixed by classifying emotion here and
     returning it.
  2. Emotion source priority: we classify the QUESTION first (a direct
     statement like "I am very happy" is the actual emotional signal),
     and only fall back to classifying the ANSWER text if the question
     doesn't carry a confident emotion. This matters because
     compound's answer text is often written in a flat, informational
     register regardless of how emotional the question was.
  3. LATENCY: default model switched from "groq/compound" to
     "groq/compound-mini". Per Groq's docs (and our own prior note in
     this file), compound-mini does a single tool call vs compound's
     multi-round tool-calling loop -- ~3x lower latency for simple
     questions, which covers the overwhelming majority of voice-avatar
     queries. Override with GROQ_WEB_SEARCH_MODEL if a specific query
     class genuinely needs compound's multi-hop research behavior.

This is a single BLOCKING call (compound's tool-calling loop -- search,
possibly multiple rounds, then synthesis -- all happens server-side before
this returns). bridge_server.py already wraps the call in
asyncio.to_thread(), so blocking here is fine and expected.
"""
import os

from groq import Groq

from backend.utils.logger import get_logger
from backend.emotion.classifier import classify_emotion

logger = get_logger(__name__)

# was "groq/compound" -- compound-mini is ~3x lower latency (single tool
# call vs compound's multi-round loop) and is sufficient for the vast
# majority of avatar questions. Set GROQ_WEB_SEARCH_MODEL=groq/compound
# explicitly if you need deep multi-hop research answers.
DEFAULT_MODEL = os.environ.get("GROQ_WEB_SEARCH_MODEL", "groq/compound-mini")

_SYSTEM_PROMPT = (
    "You are ARIA, a professional and empathetic voice assistant with access "
    "to real-time web search. Your responses will be spoken aloud by a "
    "talking avatar, so follow these formatting rules strictly: "
    "use plain spoken prose only; no markdown, no bullet points, no asterisks, "
    "no headers, and no code blocks. Keep responses concise and conversational, "
    "under 150 words unless the topic genuinely requires more depth. "
    "Never read out URLs — sources are displayed separately in the UI.\n\n"
    "EMOTION TAGGING (mandatory):\n"
    "Begin every response with exactly ONE emotion tag on its own line. "
    "This tag controls the avatar's facial expression and voice tone. "
    "Select the most contextually appropriate tag using the rules below:\n\n"
    "  [supportive]   — Select when the user appears upset, distressed, "
    "frustrated, or is expressing negative emotions. Also select for any "
    "response that is empathetic, comforting, or reassuring in nature.\n\n"
    "  [sad]          — Select for any topic involving loss of life, death, "
    "natural disasters, accidents, tragedies, violence, war, or other "
    "somber and serious subject matter. This takes priority over [professional] "
    "even when the answer is factual in tone.\n\n"
    "  [friendly]     — Select for positive, uplifting, celebratory, or "
    "light-hearted topics.\n\n"
    "  [professional] — Select only for purely technical, scientific, or "
    "business topics that contain no emotional or tragic elements.\n\n"
    "Tag selection examples:\n"
    "  Topic: a major accident or disaster           → [sad]\n"
    "  Topic: a death, tragedy, or loss of life      → [sad]\n"
    "  User expresses frustration or distress        → [supportive]\n"
    "  Topic: a celebration or good news             → [friendly]\n"
    "  Topic: a programming concept or technical API → [professional]\n"
)

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set -- required for mode='web' (same key "
                "used for RAG generation)."
            )
        _client = Groq(api_key=api_key)
    return _client


def _extract_sources(message) -> list[dict]:
    sources: list[dict] = []
    seen_urls: set[str] = set()

    executed_tools = getattr(message, "executed_tools", None) or []
    for tool_call in executed_tools:
        if isinstance(tool_call, dict):
            search_results = tool_call.get("search_results")
        else:
            search_results = getattr(tool_call, "search_results", None)

        if search_results is None:
            continue

        if isinstance(search_results, dict):
            results = search_results.get("results", [])
        elif hasattr(search_results, "results"):
            results = getattr(search_results, "results") or []
        elif isinstance(search_results, list):
            results = search_results
        else:
            results = []

        for r in results:
            if isinstance(r, dict):
                url = r.get("url") or r.get("link")
                title = r.get("title")
            else:
                url = getattr(r, "url", None) or getattr(r, "link", None)
                title = getattr(r, "title", None)

            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            sources.append({"url": url, "title": title or url})

    return sources


def _resolve_emotion(question: str, answer: str) -> str:
    """
    Classifies emotion for web-mode answers, which had NO emotion signal
    at all before this fix (the field simply didn't exist).

    Priority: the question's own emotional content first (a direct
    statement like "I am very sad" IS the signal, and compound's answer
    text often stays flat/informational even when responding to it).
    Falls back to classifying the answer body if the question reads as
    neutral (e.g. a factual lookup with no emotional content of its own).
    """
    question_emotion = classify_emotion(question)
    if question_emotion != "neutral":
        logger.info("Web emotion: from QUESTION -> %s", question_emotion)
        return question_emotion

    answer_emotion = classify_emotion(answer[:300])
    logger.info("Web emotion: question neutral, from ANSWER -> %s", answer_emotion)
    return answer_emotion


def web_search_answer(question: str, model: str | None = None) -> dict:
    """
    Runs one Groq compound chat-completion call with built-in web search
    enabled and returns the synthesized answer, source links, and a
    single resolved facial emotion for the whole answer.
    """
    client = _get_client()
    model = model or DEFAULT_MODEL

    if len(question) > 1000:
        logger.warning("Truncating web search query from %d to 1000 characters", len(question))
        question = question[:1000] + "... [truncated]"

    logger.info("Web search query (%s): %r", model, question[:120])

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    message = completion.choices[0].message
    raw_answer = (message.content or "").strip()
    sources = _extract_sources(message)
    
    import re
    match = re.match(r"^\s*\[([a-zA-Z_]+)\]\s*\n?(.*)", raw_answer, re.DOTALL)
    if match:
        emotion = match.group(1).strip().lower()
        answer_text = match.group(2).strip()
    else:
        answer_text = raw_answer
        emotion = _resolve_emotion(question, answer_text)

    logger.info(
        "Web search answer: %d chars, %d source(s), emotion=%s",
        len(answer_text), len(sources), emotion,
    )

    return {"answer": answer_text, "sources": sources, "emotion": emotion}