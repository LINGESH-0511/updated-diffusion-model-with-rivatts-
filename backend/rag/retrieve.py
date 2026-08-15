"""
backend/rag/retrieve.py

Native Qdrant hybrid search: dense (semantic) + sparse (BM25) in a single
query, fused server-side via Reciprocal Rank Fusion (RRF). This replaces
the manually-fused keyword + BM25 + dense search from the original plan.

Reranking is OFF by default (config.USE_RERANKER). Ship without it first;
flip the flag on once you've measured whether hybrid alone is good enough
for your corpus size. See ingest.py's model caching pattern if/when you
add a cross-encoder here — cache it the same way.
"""
from __future__ import annotations

import logging

from qdrant_client.http import models as qmodels

from . import config
from .ingest import get_qdrant_client, _get_dense_model, _get_sparse_model

logger = logging.getLogger(__name__)


def hybrid_search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Returns a list of {"text": str, "source": str, "score": float} dicts,
    ranked by Qdrant's server-side RRF fusion of dense + sparse results.
    """
    top_k = top_k or config.RETRIEVE_TOP_K
    client = get_qdrant_client()

    dense_vec = next(_get_dense_model().embed([query])).tolist()
    sparse_vec = next(_get_sparse_model().embed([query]))

    results = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[
            qmodels.Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k * 4,   # widen the candidate pool before fusion
            ),
            qmodels.Prefetch(
                query=qmodels.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=top_k * 4,
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    hits = [
        {
            "text": p.payload.get("text", ""),
            "source": p.payload.get("source", "unknown"),
            "score": p.score,
        }
        for p in results.points
    ]

    if config.USE_RERANKER:
        hits = _rerank(query, hits)

    return hits


def _rerank(query: str, hits: list[dict]) -> list[dict]:
    """
    Placeholder for the v2 reranking step (FastEmbed local cross-encoder,
    e.g. BAAI/bge-reranker-base). Not implemented in v1 per the "skip for
    v1" default — flip config.USE_RERANKER=True and implement this once
    you've measured that hybrid search alone isn't precise enough.
    """
    logger.warning("USE_RERANKER is True but _rerank() is not implemented yet — returning unranked hits.")
    return hits


def build_context_block(hits: list[dict], max_chars: int = 4000) -> str:
    """Concatenate retrieved chunks into a single context string for the LLM prompt."""
    parts = []
    total = 0
    for h in hits:
        chunk = f"[Source: {h['source']}]\n{h['text']}\n"
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n---\n".join(parts)
