"""
backend/rag/pipeline.py

Thin orchestration layer, kept as a single import point for bridge_server.py
(mirrors the old `from backend.rag.pipeline import upload_and_ingest, answer_query`
import line so the integration diff is small).

Call startup_check() once when bridge_server.py starts, so a dead/unreachable
Qdrant container fails loudly at boot instead of on the first user query.
"""
from __future__ import annotations

import logging

from qdrant_client.http.exceptions import ResponseHandlingException

from .ingest import upload_and_ingest, ensure_collection, get_qdrant_client
from .generate import answer_query_stream, transcribe_audio

logger = logging.getLogger(__name__)

__all__ = ["upload_and_ingest", "answer_query_stream", "transcribe_audio", "startup_check"]


def startup_check() -> None:
    """
    Verify Qdrant server mode is actually reachable before accepting any
    RAG requests. This is the server-mode equivalent of the old local-file
    lock check the plan flagged in §5.2 — fail at startup, not mid-query.
    """
    try:
        client = get_qdrant_client()
        client.get_collections()
        ensure_collection()
        logger.info("RAG startup check passed: Qdrant reachable at %s", client._client.rest_uri if hasattr(client, "_client") else "configured host")
    except ResponseHandlingException as e:
        raise RuntimeError(
            "Cannot reach Qdrant. Is the container running? "
            "Try: docker compose -f docker-compose.qdrant.yml up -d"
        ) from e
