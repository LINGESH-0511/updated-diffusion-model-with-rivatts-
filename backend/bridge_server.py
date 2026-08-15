"""
backend/bridge_server.py  -  WebSocket bridge (live avatar, CHUNKED)

Flow (per sentence chunk) -- "speak" path:
  Browser  --WS--> bridge_server            {"type":"speak","text":...}
  bridge_server  --> Riva TTS               synthesize_pcm(chunk_text, emotion)
  bridge_server  --> Browser                 {"type":"audio_header",...} + wav bytes
  bridge_server  --> A2F NIM                 stream_animation_frames(pcm, sr, config, preferred_emotion)
  bridge_server  --> Browser (per frame)     {"type":"blendshapes","chunkId":N,"frameIndex":i,...}
  bridge_server  --> Browser (chunk done)    {"type":"chunk_end","chunkId":N,"totalFrames":i,...}
  ... repeat for chunk N+1 ...
  bridge_server  --> Browser (all done)      {"type":"all_done","totalChunks":M}

Flow -- "ask" path (RAG or Web Search, chosen by the client's "mode" field):
  Browser  --WS--> bridge_server            {"type":"upload_document","filename":...,"reset":false} + binary frame
  bridge_server  --> Browser (per stage)     {"type":"upload_progress","message":...}
  bridge_server  --> Qdrant                  ingest_document() via backend.rag.ingest
  bridge_server  --> Browser                 {"type":"upload_done","chunks":N}

  Browser  --WS--> bridge_server            {"type":"ask","question":...,"mode":"rag"|"web"}
    mode == "rag" (default):
      bridge_server --> Qdrant               hybrid_search()
      bridge_server --> Groq                 answer_query_stream() -- STREAMED tokens
      bridge_server --> (same TTS/A2F pipeline as "speak")
      NOTE: RAG mode now uses ONE emotion per whole answer (same as web
      mode) -- generate.py's system prompt was updated this session to
      emit a single leading [emotion] tag before the entire answer.
      The tag is parsed once in _handle_ask(), stripped, and that single
      emotion is applied to every chunk, keeping both TTS prosody and
      A2F facial expression consistent across the whole answer.
    mode == "web" (single blocking call, ONE emotion for the whole answer):
      bridge_server --> Groq (compound)      backend.websearch.web_search_answer()
      bridge_server --> Browser              {"type":"sources","sources":[...]}  (if any found)
      bridge_server --> (same TTS/A2F pipeline as "speak", every chunk of the answer
                          uses that single resolved emotion)
  bridge_server  --> Browser                 same audio_header/blendshapes/chunk_end/all_done messages

Flow -- "voice_ask" path:
  Browser  --WS--> bridge_server            {"type":"voice_ask","mode":"rag"|"web"} + binary audio frame
  bridge_server  --> Groq                    transcribe_audio_bytes() (Whisper STT)
  bridge_server  --> Browser                 {"type":"transcript","text":...}
  bridge_server  --> (same "ask" pipeline from here)

Run from D:\\A2F:
    python -m backend.bridge_server

Browser connects to:  ws://localhost:8765

IMPORTANT: restart this process every time you edit this file.

REQUIRES for the "ask"/"upload_document"/"voice_ask" paths:
  - Qdrant running:            docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
  - GROQ_API_KEY set in environment
  - backend/rag/ package present

TTS backend (Riva/Magpie):
  - riva_streamer.py hits the NVIDIA API Catalog REST endpoint at startup.
  - synthesize_pcm(text, emotion) has the same signature as before;
    emotion selects a speech-rate preset that maps onto Riva's speed param.

Emotion → A2F wiring:
  - _emotion_to_preferred_emotion() converts the resolved emotion string
    into the dict[str, float] shape stream.send_audio() expects.
  - stream.py (_build_emotion_timecode_list) then expands that sparse
    dict into a FULL 6-key vector before it reaches the gRPC call, so
    A2F receives explicit weights for all 6 emotions rather than a
    partial dict it might silently ignore.
  - "neutral" passes preferred_emotion=None, deliberately leaving A2E's
    own audio-based inference in control for that chunk.
  - A2F only accepts SIX values: anger, disgust, fear, joy, neutral,
    sadness. VALID_EMOTIONS and _EMOTION_REMAP are constrained to these.
"""
import os

if os.name == "nt":
    _cudnn_bin = r"E:\a2f_env\Lib\site-packages\nvidia\cudnn\bin"
    _cublas_bin = r"E:\a2f_env\Lib\site-packages\nvidia\cublas\bin"
    if os.path.isdir(_cudnn_bin):
        os.add_dll_directory(_cudnn_bin)
    if os.path.isdir(_cublas_bin):
        os.add_dll_directory(_cublas_bin)

import asyncio
import io
import json
import logging
import re
import tempfile
import time
import wave

import websockets

from backend.audio2face.receiver_streaming import stream_animation_frames
from backend.audio2face.client import get_shared_stub, close_shared_channel
from backend.tts.riva_streamer import synthesize_pcm as riva_synthesize_pcm, preload_voice as riva_preload_voice
from backend.tts.piper_streamer import synthesize_pcm as piper_synthesize_pcm, preload_voice as piper_preload_voice
from backend.audio2face.wav_sender import load_config
from backend.utils.logger import get_logger
from backend.rag.pipeline import answer_query_stream, startup_check
from backend.rag.ingest import upload_and_ingest, preload_models
from backend.rag.stt import transcribe_audio_bytes, MAX_AUDIO_BYTES
from backend.websearch import web_search_answer
from backend.emotion.classifier import classify_emotion, warm_up_classifier

logger = get_logger(__name__)

def synthesize_pcm(text: str, emotion: str = "neutral") -> tuple[bytes, int, str]:
    try:
        pcm, sr = riva_synthesize_pcm(text, emotion)
        return pcm, sr, "riva"
    except Exception as e:
        logger.warning("Riva TTS failed: %s. Falling back to Piper TTS.", e)
        pcm, sr = piper_synthesize_pcm(text, emotion)
        return pcm, sr, "piper"

def preload_voice():
    try:
        riva_preload_voice()
    except Exception as e:
        logger.warning("Riva TTS preload failed: %s", e)
    piper_preload_voice()

CONFIG_PATH = os.environ.get(
    "A2F_CONFIG",
    "Audio2Face-3D-Samples/scripts/audio2face_3d_microservices_interaction_app/config/config_james.yml",
)

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\s+")

# --- Chunk sizing constants (latency-tuned) --------------------------------
_CHUNK_FIRST_MAX_WORDS = 3
_CHUNK_MAX_WORDS = 15

# Max bytes accepted for a single document upload (50MB).
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Valid values for the client-supplied "mode" field on "ask"/"voice_ask".
_VALID_ASK_MODES = {"rag", "web"}

# --- Emotion tag convention -------------------------------------------------
# Two vocabularies combined:
#   1. LLM-facing "voice tone" tags: friendly / caring / professional / neutral
#      These drive TTS prosody and get REMAPPED onto A2F's facial set below.
#   2. A2F's real facial emotion set (6 values only):
#      anger, disgust, fear, joy, neutral, sadness
VALID_EMOTIONS = {
    "joy", "sadness", "neutral",
    "friendly", "supportive", "professional", "sad",
}

# Maps LLM-facing tags onto A2F's real 6-value facial emotion set.
_EMOTION_REMAP = {
    "friendly":     "joy",
    "supportive":   "sadness",   # concerned/empathetic look, not a smile
    "professional": "neutral",
    "sad":          "sadness",
    "neutral":      "neutral",
    "joy":          "joy",
    "sadness":      "sadness",
}

# Matches a leading "[emotion]" tag at the start of a sentence.
_EMOTION_TAG_RE = re.compile(r"^\s*\[([a-zA-Z_]+)\]\s*(.*)$", re.DOTALL)

# Matches a leading "[emotion]" tag on its own line at the start of a
# full answer block (used by RAG mode's single-tag-per-answer parsing).
_LEADING_TAG_RE = re.compile(r"^\s*\[([a-zA-Z_]+)\]\s*\n?(.*)", re.DOTALL)


def _extract_emotion_tag(sentence: str) -> tuple[str, str]:
    """
    Strips a leading "[emotion] " tag off a single sentence, if present.
    Returns (emotion, clean_text). Always returns a valid VALID_EMOTIONS value.
    """
    match = _EMOTION_TAG_RE.match(sentence)
    if not match:
        return "neutral", sentence.strip()

    tag = match.group(1).strip().lower()
    clean_text = match.group(2).strip()

    if tag not in VALID_EMOTIONS:
        logger.warning("Unrecognized emotion tag %r, falling back to 'neutral'", tag)
        tag = "neutral"

    if not clean_text:
        return "neutral", sentence.strip()

    return tag, clean_text


def _extract_leading_answer_emotion(text: str) -> tuple[str, str]:
    """
    Parses a single leading [emotion] tag that appears before the entire
    answer body (one tag for the whole response, not per sentence).

    Used by RAG mode (_handle_ask) which now uses the same single-tag
    convention as web-search mode. Falls back to "neutral" + original
    text if no leading tag is found.

    Returns (emotion, answer_text_without_tag).
    """
    match = _LEADING_TAG_RE.match(text)
    if not match:
        logger.warning("RAG answer had no leading [emotion] tag — defaulting to 'neutral'")
        return "neutral", text.strip()

    tag = match.group(1).strip().lower()
    answer_body = match.group(2).strip()

    if tag not in VALID_EMOTIONS:
        logger.warning("RAG answer leading tag %r not recognized — defaulting to 'neutral'", tag)
        tag = "neutral"

    logger.info("RAG answer emotion (single tag): %s", tag)
    return tag, answer_body


def split_into_chunks(text: str) -> list[str]:
    """
    Plain word-chunking, no emotion. Used by the web-search path and now
    also by the RAG path (which resolves one emotion up front).
    """
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
    if not sentences:
        sentences = [text]

    chunks: list[str] = []
    for sentence in sentences:
        words = [w for w in _WORD_RE.split(sentence) if w]
        pos = 0
        while pos < len(words):
            limit = _CHUNK_FIRST_MAX_WORDS if not chunks else _CHUNK_MAX_WORDS
            piece = " ".join(words[pos:pos + limit])
            if piece:
                chunks.append(piece)
            pos += limit

    return chunks


def split_into_chunks_with_emotion(text: str) -> tuple[list[str], list[str]]:
    """
    Emotion-aware chunker. Used only by the "speak" path now.
    RAG mode has been moved to single-tag-per-answer (see _handle_ask).

    If a sentence has NO explicit "[emotion]" tag, it is run through
    the emotion classifier instead of silently defaulting to "neutral".

    Returns (chunks, emotions) -- same length, index-aligned.
    """
    text = text.strip()
    if not text:
        return [], []

    raw_sentences = [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]
    if not raw_sentences:
        raw_sentences = [text]

    chunks: list[str] = []
    emotions: list[str] = []

    for raw_sentence in raw_sentences:
        emotion, clean_sentence = _extract_emotion_tag(raw_sentence)

        if emotion == "neutral" and not _EMOTION_TAG_RE.match(raw_sentence):
            emotion = classify_emotion(clean_sentence)
            logger.info("CLASSIFIER: %r -> emotion=%s", clean_sentence[:80], emotion)

        words = [w for w in _WORD_RE.split(clean_sentence) if w]
        pos = 0
        while pos < len(words):
            limit = _CHUNK_FIRST_MAX_WORDS if not chunks else _CHUNK_MAX_WORDS
            piece = " ".join(words[pos:pos + limit])
            if piece:
                chunks.append(piece)
                emotions.append(emotion)
            pos += limit

    return chunks, emotions


def pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


# When overriding A2E audio inference with explicit tags (e.g. from Piper fallback),
# this is the weight applied to the chosen emotion.
_PREFERRED_EMOTION_WEIGHT = 1.0


def _emotion_to_preferred_emotion(emotion: str) -> dict[str, float] | None:
    """
    Converts the resolved emotion string into the dict shape stream.send_audio() expects.
    "neutral" passes preferred_emotion=None to allow A2E's audio inference to control.
    """
    mapped = _EMOTION_REMAP.get(emotion, emotion)
    if mapped == "neutral":
        logger.info("A2F EMOTION: %s -> mapped to neutral -> None", emotion)
        return None
        
    logger.info("A2F EMOTION: %s -> mapped to %s (weight: %s)", emotion, mapped, _PREFERRED_EMOTION_WEIGHT)
    return {mapped: _PREFERRED_EMOTION_WEIGHT}


async def _speak_chunks(ws, chunks: list[str], chunk_id_offset: int = 0,
                         emotions: list[str] | None = None) -> int:
    """
    Runs the TTS -> A2F -> WebSocket pipeline over a list of text chunks.
    Shared by the "speak", "ask" (RAG), and "ask" (web search) paths.
    """
    total_chunks = len(chunks)
    if total_chunks == 0:
        return 0

    if emotions is None:
        emotions = ["neutral"] * total_chunks
    elif len(emotions) != total_chunks:
        logger.warning("emotions list length (%d) != chunks length (%d), padding with 'neutral'",
                        len(emotions), total_chunks)
        emotions = (emotions + ["neutral"] * total_chunks)[:total_chunks]

    t_request_start = time.monotonic()
    config = load_config(CONFIG_PATH)

    tts_tasks: dict[int, asyncio.Task] = {}

    def ensure_tts_task(i: int):
        if 0 <= i < len(chunks) and i not in tts_tasks:
            tts_tasks[i] = asyncio.create_task(
                asyncio.to_thread(synthesize_pcm, chunks[i], emotions[i])
            )

    ensure_tts_task(0)

    spoken = 0
    for local_id, chunk_text in enumerate(chunks):
        if getattr(ws, 'stop_requested', False):
            logger.info('STOP requested during playback. Cutting off audio.')
            ws.stop_requested = False
            try: await ws.send(json.dumps({'type': 'stopped'}))
            except Exception: pass
            break
            
        chunk_id = local_id + chunk_id_offset
        emotion = emotions[local_id]
        t_chunk_start = time.monotonic()

        ensure_tts_task(local_id + 1)

        try:
            pcm_bytes, sample_rate, tts_engine = await tts_tasks[local_id]
        except Exception as e:
            logger.exception("Riva TTS failed on chunk %d", chunk_id)
            await ws.send(json.dumps({
                "type": "error",
                "message": f"TTS error on chunk {chunk_id}: {e}",
            }))
            break

        t_tts_done = time.monotonic()
        logger.info("TIMING chunk %d: TTS ready after %.2fs wait - %d bytes @ %d Hz - emotion=%s - %r",
                    chunk_id, t_tts_done - t_chunk_start,
                    len(pcm_bytes), sample_rate, emotion, chunk_text[:60])

        await ws.send(json.dumps({
            "type": "response_text",
            "chunkId": chunk_id,
            "text": chunk_text,
            "emotion": emotion,
        }))

        wav_bytes = pcm_to_wav_bytes(pcm_bytes, sample_rate)
        await ws.send(json.dumps({
            "type": "audio_header",
            "chunkId": chunk_id,
            "totalChunks": total_chunks,
            "byteLength": len(wav_bytes),
        }))
        await ws.send(wav_bytes)
        logger.info("TIMING chunk %d: audio sent at +%.2fs since request start",
                    chunk_id, time.monotonic() - t_request_start)

        preferred_emotion = _emotion_to_preferred_emotion(emotion)
        logger.info("A2F EMOTION CHECK: chunk %d emotion=%s -> preferred_emotion=%s (Explicit weights used to guarantee facial expressiveness for %s)", chunk_id, emotion, preferred_emotion, tts_engine)

        total_frames = 0
        status_code = None
        status_message = ""
        t_first_frame = None
        try:
            async for frame in stream_animation_frames(
                pcm_bytes, sample_rate, config, preferred_emotion=preferred_emotion
            ):
                if frame["type"] == "blendshapes":
                    if t_first_frame is None:
                        t_first_frame = time.monotonic()
                        logger.info(
                            "TIMING chunk %d: first A2F frame after %.2fs",
                            chunk_id, t_first_frame - t_tts_done,
                        )
                    frame["chunkId"] = chunk_id
                    await ws.send(json.dumps(frame))
                    total_frames += 1
                elif frame["type"] == "end":
                    status_code = frame.get("statusCode")
                    status_message = frame.get("statusMessage", "")
        except Exception as e:
            logger.exception("A2F streaming error on chunk %d", chunk_id)
            await ws.send(json.dumps({
                "type": "error",
                "message": f"A2F error on chunk {chunk_id}: {e}",
            }))
            break

        t_chunk_done = time.monotonic()
        logger.info("TIMING chunk %d: TOTAL %.2fs (TTS %.2fs, A2F %.2fs), %d frames",
                    chunk_id, t_chunk_done - t_chunk_start,
                    t_tts_done - t_chunk_start, t_chunk_done - t_tts_done,
                    total_frames)

        await ws.send(json.dumps({
            "type": "chunk_end",
            "chunkId": chunk_id,
            "totalFrames": total_frames,
            "statusCode": status_code,
            "statusMessage": status_message,
        }))
        spoken += 1

    return spoken


async def _iter_in_thread(sync_generator_fn, *args):
    """
    Bridges a blocking/synchronous generator into an async generator,
    using a background-thread + queue.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _run():
        try:
            for item in sync_generator_fn(*args):
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    import threading
    threading.Thread(target=_run, daemon=True).start()

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def _handle_ask(ws, question: str):
    """
    RAG query path (mode == "rag").

    UPDATED this session: now uses ONE emotion for the whole answer,
    matching the web-search path. generate.py's system prompt now emits
    exactly ONE leading [emotion] tag before the entire answer body.
    We collect all streamed tokens first, parse the single leading tag,
    strip it, then speak the answer with that emotion applied to every
    chunk.

    Trade-off accepted: this means we wait for the full answer before
    starting TTS, losing the "start speaking before generation finishes"
    latency win. The benefit is consistent facial expression and prosody
    across the whole answer instead of per-sentence mood swings.

    If you want to restore the streaming-first approach, revert to the
    per-sentence tag path and update generate.py's system prompt back to
    per-sentence tags.
    """
    t_start = time.monotonic()
    logger.info("Ask request (RAG): %r", question[:120])

    # Collect the full answer first so we can parse the single leading tag.
    full_answer = ""
    got_any_tokens = False

    try:
        async for token in _iter_in_thread(answer_query_stream, question):
            if getattr(ws, "stop_requested", False):
                ws.stop_requested = False
                break
            got_any_tokens = True
            full_answer += token
            # Send token so it prints while waiting for full answer!
            await ws.send(json.dumps({"type": "response_text", "text": token}))
    except Exception as e:
        logger.exception("RAG ask failed")
        await ws.send(json.dumps({"type": "error", "message": f"Ask error: {e}"}))
        return

    if not got_any_tokens or not full_answer.strip():
        await ws.send(json.dumps({
            "type": "error",
            "message": "No answer generated (no documents ingested yet?).",
        }))
        return

    # Parse the single leading [emotion] tag emitted by generate.py's
    # updated system prompt.
    emotion, answer_body = _extract_leading_answer_emotion(full_answer)

    if not answer_body.strip():
        await ws.send(json.dumps({
            "type": "error",
            "message": "Answer body was empty after stripping emotion tag.",
        }))
        return

    logger.info(
        "RAG answer collected in %.2fs: emotion=%s, length=%d chars",
        time.monotonic() - t_start, emotion, len(answer_body),
    )

    # Every chunk of this answer shares the same emotion — consistent
    # prosody and facial expression across the whole response.
    chunks = split_into_chunks(answer_body)
    emotions = [emotion] * len(chunks)
    spoken = await _speak_chunks(ws, chunks, emotions=emotions)

    logger.info("Ask request (RAG) done in %.2fs, %d chunk(s) spoken, emotion=%s",
                time.monotonic() - t_start, spoken, emotion)
    await ws.send(json.dumps({
        "type": "all_done",
        "totalChunks": spoken,
    }))


async def _handle_web_ask(ws, question: str):
    """
    Web-search query path (mode == "web"). Single blocking Groq compound
    call, then speaks the answer with ONE emotion for the entire response.
    """
    t_start = time.monotonic()
    logger.info("Ask request (web search): %r", question[:120])

    try:
        result = await asyncio.to_thread(web_search_answer, question)
    except Exception as e:
        logger.exception("Web search ask failed")
        err_str = str(e)
        if "413" in err_str or "too_large" in err_str:
            await ws.send(json.dumps({"type": "error", "message": "The web search retrieved too much information for this topic. Please ask a more specific question, or use Document mode."}))
        else:
            await ws.send(json.dumps({"type": "error", "message": f"Web search error: {e}"}))
        return

    answer_text = (result.get("answer") or "").strip()
    emotion = result.get("emotion") or "neutral"
    sources = result.get("sources") or []

    if not answer_text:
        await ws.send(json.dumps({
            "type": "error",
            "message": "No answer returned from web search.",
        }))
        return

    if emotion not in VALID_EMOTIONS:
        logger.warning("web_search_answer returned unrecognized emotion %r, falling back to 'neutral'", emotion)
        emotion = "neutral"

    if sources:
        await ws.send(json.dumps({"type": "sources", "sources": sources}))

    chunks = split_into_chunks(answer_text)
    emotions = [emotion] * len(chunks)
    spoken = await _speak_chunks(ws, chunks, emotions=emotions)

    if spoken == 0:
        await ws.send(json.dumps({
            "type": "error",
            "message": "Web search answer could not be spoken (TTS/A2F failed on first chunk).",
        }))
        return

    logger.info("Ask request (web search) done in %.2fs, %d chunk(s) spoken, emotion=%s, %d source(s)",
                time.monotonic() - t_start, spoken, emotion, len(sources))
    await ws.send(json.dumps({
        "type": "all_done",
        "totalChunks": spoken,
    }))


async def _dispatch_ask(ws, question: str, mode: str):
    ws.stop_requested = False
    if mode == "web":
        await _handle_web_ask(ws, question)
    else:
        await _handle_ask(ws, question)


def _normalize_mode(raw_mode) -> str:
    mode = (raw_mode or "rag")
    if mode not in _VALID_ASK_MODES:
        logger.warning("Unknown ask mode %r, falling back to 'rag'", raw_mode)
        mode = "rag"
    return mode


async def handle_client(ws):
    addr = ws.remote_address
    logger.info("Client connected: %s", addr)

    pending_upload: dict | None = None
    pending_voice_mode: str | None = None
    current_task: asyncio.Task | None = None

    loop = asyncio.get_event_loop()

    ws.stop_requested = False

    async def _cancel_current():
        nonlocal current_task
        if current_task and not current_task.done():
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                pass
        current_task = None
        
    async def _request_stop():
        ws.stop_requested = True

    async def _task_wrapper(coro):
        try:
            await coro
        except asyncio.CancelledError:
            pass

    def make_progress_cb():
        def _cb(message: str):
            asyncio.run_coroutine_threadsafe(
                ws.send(json.dumps({"type": "upload_progress", "message": message})),
                loop,
            )
        return _cb

    try:
        async for raw_msg in ws:
            if isinstance(raw_msg, (bytes, bytearray)):
                if pending_upload is not None:
                    if len(raw_msg) > _MAX_UPLOAD_BYTES:
                        await ws.send(json.dumps({
                            "type": "error",
                            "message": f"Upload too large ({len(raw_msg)} bytes, max {_MAX_UPLOAD_BYTES})",
                        }))
                        pending_upload = None
                        continue

                    filename = pending_upload["filename"]
                    reset = pending_upload.get("reset", False)
                    suffix = os.path.splitext(filename)[1] or ".bin"
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(raw_msg)
                            tmp_path = tmp.name

                        logger.info("Ingesting upload '%s' (%d bytes, reset=%s)", filename, len(raw_msg), reset)
                        t0 = time.monotonic()
                        n_chunks = await asyncio.to_thread(
                            upload_and_ingest, tmp_path, filename, reset, make_progress_cb()
                        )
                        await ws.send(json.dumps({
                            "type": "upload_done",
                            "filename": filename,
                            "chunks": n_chunks,
                        }))
                        logger.info("Ingested '%s' -> %d chunks in %.2fs", filename, n_chunks, time.monotonic() - t0)
                    except Exception as e:
                        logger.exception("Ingestion failed for '%s'", filename)
                        await ws.send(json.dumps({
                            "type": "error",
                            "message": f"Ingestion failed: {e}",
                        }))
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                        pending_upload = None
                    continue

                elif pending_voice_mode is not None:
                    mode = pending_voice_mode
                    pending_voice_mode = None

                    if len(raw_msg) > MAX_AUDIO_BYTES:
                        await ws.send(json.dumps({
                            "type": "error",
                            "message": f"Audio too large ({len(raw_msg)} bytes, max {MAX_AUDIO_BYTES})",
                        }))
                        continue

                    try:
                        t0 = time.monotonic()
                        transcript = await asyncio.to_thread(transcribe_audio_bytes, raw_msg)
                        logger.info("Voice transcribed in %.2fs: %r", time.monotonic() - t0, transcript[:120])
                    except Exception as e:
                        logger.exception("Voice transcription failed")
                        await ws.send(json.dumps({
                            "type": "error",
                            "message": f"Transcription error: {e}",
                        }))
                        continue

                    if not transcript:
                        await ws.send(json.dumps({
                            "type": "error",
                            "message": "Transcription returned empty text -- try speaking again.",
                        }))
                        continue

                    await ws.send(json.dumps({"type": "transcript", "text": transcript}))
                    await _cancel_current()
                    current_task = asyncio.create_task(_task_wrapper(_dispatch_ask(ws, transcript, mode)))
                    continue

                else:
                    await ws.send(json.dumps({
                        "type": "error",
                        "message": "Received binary data with no pending upload_document or voice_ask message",
                    }))
                    continue

            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                await ws.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            msg_type = msg.get("type")

            if msg_type == "speak":
                text = msg.get("text", "").strip()
                if not text:
                    await ws.send(json.dumps({"type": "error", "message": "Empty text"}))
                    continue

                chunks, emotions = split_into_chunks_with_emotion(text)
                logger.info("Speak request: %r -> %d chunk(s)", text[:80], len(chunks))
                async def _speak_task():
                    ws.stop_requested = False
                    await _speak_chunks(ws, chunks, emotions=emotions)
                    await ws.send(json.dumps({"type": "all_done", "totalChunks": len(chunks)}))
                await _cancel_current()
                current_task = asyncio.create_task(_task_wrapper(_speak_task()))

            elif msg_type == "upload_document":
                filename = msg.get("filename")
                if not filename:
                    await ws.send(json.dumps({"type": "error", "message": "Missing filename"}))
                    continue
                pending_upload = {"filename": filename, "reset": bool(msg.get("reset", False))}
                logger.info("Awaiting binary frame for upload: %s (reset=%s)", filename, pending_upload["reset"])

            elif msg_type == "voice_ask":
                pending_voice_mode = _normalize_mode(msg.get("mode"))
                logger.info("Awaiting binary frame for voice input (mode=%s)", pending_voice_mode)

            elif msg_type == "stop":
                logger.info("Received STOP request from client.")
                await _cancel_current()

            elif msg_type == "ask":
                question = msg.get("question", "").strip()
                if not question:
                    await ws.send(json.dumps({"type": "error", "message": "Empty question"}))
                    continue
                mode = _normalize_mode(msg.get("mode"))
                await _cancel_current()
                current_task = asyncio.create_task(_task_wrapper(_dispatch_ask(ws, question, mode)))

            else:
                await ws.send(json.dumps({"type": "error", "message": "Unknown message type"}))

    except websockets.exceptions.ConnectionClosedOK:
        logger.info("Client disconnected cleanly: %s", addr)
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning("Client disconnected with error: %s - %s", addr, e)
    except Exception:
        logger.exception("Unhandled error for client %s", addr)


async def warm_up():
    t0 = time.monotonic()
    try:
        await asyncio.to_thread(preload_voice)
        logger.info("Warm-up: Riva TTS warm-up completed in %.2fs", time.monotonic() - t0)
    except Exception:
        logger.exception("Warm-up: Riva TTS warm-up failed (non-fatal, continuing)")

    t0b = time.monotonic()
    try:
        await asyncio.to_thread(warm_up_classifier)
        logger.info("Warm-up: emotion classifier loaded in %.2fs", time.monotonic() - t0b)
    except Exception:
        logger.exception("Warm-up: emotion classifier load failed (non-fatal, continuing)")

    t1 = time.monotonic()
    try:
        await get_shared_stub()
        logger.info("Warm-up: A2F gRPC channel opened in %.2fs", time.monotonic() - t1)
    except Exception:
        logger.exception("Warm-up gRPC connect failed (non-fatal, continuing)")
        return

    config = load_config(CONFIG_PATH)
    for i in range(2):
        t2 = time.monotonic()
        try:
            pcm_bytes, sample_rate = await asyncio.to_thread(
                synthesize_pcm, "This is a warm up sentence for the model.", "neutral"
            )
            frame_count = 0
            async for frame in stream_animation_frames(pcm_bytes, sample_rate, config):
                if frame["type"] == "blendshapes":
                    frame_count += 1
            logger.info("Warm-up: A2F real inference #%d took %.2fs (%d frames)",
                        i + 1, time.monotonic() - t2, frame_count)
        except Exception:
            logger.exception("Warm-up A2F inference #%d failed (non-fatal, continuing)", i + 1)
            break

    t3 = time.monotonic()
    try:
        await asyncio.to_thread(preload_models)
        logger.info("Warm-up: RAG embedding models (dense+sparse) loaded in %.2fs", time.monotonic() - t3)
    except Exception:
        logger.exception("Warm-up: RAG embedding model preload FAILED (non-fatal, continuing)")

    try:
        await asyncio.to_thread(startup_check)
        logger.info("Warm-up: RAG/Qdrant startup check passed")
    except Exception:
        logger.exception("Warm-up: RAG/Qdrant startup check FAILED (non-fatal, continuing)")


async def main():
    host, port = "localhost", 8765
    await warm_up()
    logger.info("WebSocket bridge starting on ws://%s:%d", host, port)
    try:
        async with websockets.serve(handle_client, host, port, max_size=None):
            await asyncio.Future()
    finally:
        await close_shared_channel()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    asyncio.run(main())