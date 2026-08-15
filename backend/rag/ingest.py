"""
backend/rag/ingest.py

Document upload -> text extraction -> chunking -> embedding -> Qdrant storage.

Stores BOTH a dense vector (FastEmbed/BGE, ONNX) and a sparse vector (BM25)
per chunk, in ONE Qdrant collection, so retrieval.py can do native hybrid
search in a single call instead of running two separate searches and fusing
them in Python.

Changes in this revision (upload-speed pass):
  - preload_models(): call once during server warm-up so the first real
    upload doesn't pay ONNX model-init/download cost mid-request.
  - Dense and sparse embedding now run concurrently on a small thread pool
    instead of sequentially (both release the GIL during ONNX inference).
  - reset_first now defaults to False -- ingest_document() no longer wipes
    the whole collection on every single upload. Pass reset=True explicitly
    (e.g. from a "start fresh" action) when you actually want that.
  - PDF extraction switched from pypdf to PyMuPDF (fitz), which is
    noticeably faster on longer documents. Requires:
        pip install pymupdf --break-system-packages
  - Optional progress_cb(str) parameter so callers (bridge_server.py) can
    surface "extracting / chunking / embedding / storing" to the client
    instead of a single opaque wait.

Changes in revision (speed pass, part 2):
  - Qdrant upsert now defaults to wait=False. The previous wait=True made
    ingest_document() block until Qdrant confirmed the write was durable
    on disk, which was a real, measurable chunk of total ingest time for
    no benefit on a local instance. Exposed as ingest_document(...,
    wait_for_durability=...) so callers can opt back into wait=True for
    any flow that truly needs a synchronous durability guarantee.
    NOTE: because retrieval can now (in theory) race a write that hasn't
    finished indexing, if you ever see a freshly-uploaded document not
    show up in an immediate query, that's the first thing to suspect --
    pass wait_for_durability=True for that call site instead of reverting
    this globally.

Changes in revision (speed pass, part 3 -- ONNX thread contention):
  - _onnx_providers_kwargs() no longer requires a manual config value to
    do anything useful. Previously, on a machine with no
    config.ONNX_INTRA_OP_THREADS set, each of the two concurrent ONNX
    sessions (dense + sparse, launched together in _embed_all) would try
    to claim every available CPU core for itself by default -- on a
    CPU-constrained box (laptop, e.g. an RTX 4050 Laptop rig) this means
    the two "concurrent" embedding calls actually contend for cores
    instead of truly overlapping, silently eating some of the win the
    ThreadPoolExecutor change was supposed to buy.
    Now: if config.ONNX_INTRA_OP_THREADS is unset (None, the default),
    and the machine has more than 2 cores, each session is automatically
    capped to half the available cores (leaving room for the other
    session to run alongside it without starving it). On <=2 core
    machines the cap is skipped entirely (no ONNX default override) since
    splitting 1-2 cores in half would likely hurt more than help.
    config.ONNX_INTRA_OP_THREADS still works exactly as before for a
    manual override:
      - unset / not present on config          -> new auto-cap behavior
      - set to a positive int (e.g. 2)          -> use that value exactly
      - set to 0                                -> explicitly disable
                                                    capping, use ONNX's own
                                                    default (old behavior)
    This is a heuristic, not a guaranteed win -- if a benchmark on your
    hardware shows a different split works better, set
    config.ONNX_INTRA_OP_THREADS explicitly to override the heuristic.

Changes in THIS revision (speed pass, part 4 -- GPU embedding):
  - _onnx_providers_kwargs() now prefers CUDAExecutionProvider, with
    CPUExecutionProvider as an automatic fallback if CUDA can't be used
    for a given session.
  - IMPORTANT, and the actual root cause of an earlier failed attempt at
    this: passing providers=["CUDAExecutionProvider", ...] to FastEmbed's
    TextEmbedding/SparseTextEmbedding constructors silently does nothing
    unless the `fastembed-gpu` package is installed instead of the plain
    `fastembed` package. The two packages cannot coexist in the same
    environment (same rule as onnxruntime vs onnxruntime-gpu) -- the
    plain package has to be uninstalled first:
        pip uninstall fastembed -y
        pip install fastembed-gpu
    Without this, FastEmbed reports CUDAExecutionProvider as "available"
    (because onnxruntime-gpu is genuinely installed) but the embedding
    calls still run on CPU with no error or warning -- confirmed via
    _log_actual_provider() below, which introspects the model's actual
    active provider list rather than trusting the requested one.
  - config.USE_CUDA_EMBEDDING = False forces CPU-only (e.g. to keep the
    GPU free for the A2F NIM, which may share the same device) without
    editing this function.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from . import config

logger = logging.getLogger(__name__)

ProgressCB = Callable[[str], None] | None

# ---- Module-level caches (load once per process). ----
_dense_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None
_client: QdrantClient | None = None

# Logged once so it's easy to confirm from the startup log which thread
# cap (if any) is actually in effect, instead of having to infer it.
_onnx_thread_choice_logged = False


def _resolve_onnx_intra_op_threads() -> int | None:
    """
    Returns the intra-op thread cap to use for each ONNX session, or None
    to leave ONNX Runtime's own default behavior untouched.

    Precedence:
      1. config.ONNX_INTRA_OP_THREADS explicitly set to a positive int
         -> use it exactly.
      2. config.ONNX_INTRA_OP_THREADS explicitly set to 0
         -> explicitly disabled, return None (old/default ONNX behavior).
      3. config.ONNX_INTRA_OP_THREADS unset/None (the common case)
         -> auto-cap to half the CPU cores when there are more than 2,
            so the concurrent dense+sparse embedding calls in _embed_all
            don't contend with each other for every core. Skipped on
            <=2 core machines.
    """
    global _onnx_thread_choice_logged

    configured = getattr(config, "ONNX_INTRA_OP_THREADS", None)

    if configured is not None:
        chosen = configured if configured > 0 else None
        source = "config.ONNX_INTRA_OP_THREADS (explicit)"
    else:
        cpu_count = os.cpu_count() or 4
        if cpu_count > 2:
            chosen = max(1, cpu_count // 2)
        else:
            chosen = None
        source = f"auto (cpu_count={cpu_count})"

    if not _onnx_thread_choice_logged:
        logger.info(
            "ONNX intra-op thread cap: %s [source: %s]",
            chosen if chosen else "unset (ONNX default)", source,
        )
        _onnx_thread_choice_logged = True

    return chosen


def _onnx_providers_kwargs() -> dict:
    """
    Builds the ONNX Runtime provider kwargs for each embedding session.

    Prefers CUDAExecutionProvider (confirmed available on this machine via
    `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`)
    with CPUExecutionProvider listed second as an automatic fallback --
    ONNX Runtime tries providers in list order and falls through silently
    if the first one can't be used for a given session (e.g. a transient
    VRAM shortage from the A2F NIM sharing the same GPU), so ingestion
    degrades to CPU instead of failing outright.

    config.ONNX_INTRA_OP_THREADS still applies to the CPU provider's
    thread count (relevant on fallback, and for the CPU path in general).
    It has no effect on the CUDA provider, which does its own internal
    scheduling.

    Set config.USE_CUDA_EMBEDDING = False to force CPU-only (e.g. if you
    want to keep the GPU free for A2F NIM exclusively) without touching
    this function.
    """
    use_cuda = getattr(config, "USE_CUDA_EMBEDDING", True)
    threads = _resolve_onnx_intra_op_threads()

    cpu_provider = (
        "CPUExecutionProvider",
        {"intra_op_num_threads": threads, "inter_op_num_threads": 1},
    ) if threads else "CPUExecutionProvider"

    if use_cuda:
        return {"providers": ["CUDAExecutionProvider", cpu_provider]}

    if not threads:
        return {}
    return {"providers": [cpu_provider]}


def _log_actual_provider(label: str, model) -> None:
    """
    _onnx_providers_kwargs() states a PREFERENCE (e.g. try CUDA, fall back
    to CPU) -- it does not guarantee which provider ends up active. Log
    what actually got selected so a silent CUDA->CPU fallback (e.g. from a
    VRAM shortage, or the plain `fastembed` package being installed
    instead of `fastembed-gpu` -- see module docstring) shows up in the
    log instead of looking identical to a successful GPU load.

    Attribute path confirmed from FastEmbed's own GPU usage docs:
    `embedding_model.model.model.get_providers()`.
    """
    try:
        providers = model.model.model.get_providers()
        logger.info("%s active ONNX providers: %s", label, providers)
    except Exception:
        logger.info("%s: provider introspection failed (non-fatal, cosmetic only)", label)


def _get_dense_model() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        logger.info("Loading dense embedding model: %s", config.DENSE_EMBED_MODEL)
        t0 = time.monotonic()
        _dense_model = TextEmbedding(
            model_name=config.DENSE_EMBED_MODEL,
            **_onnx_providers_kwargs(),
        )
        logger.info("Dense embedding model loaded in %.2fs", time.monotonic() - t0)
        _log_actual_provider("Dense model", _dense_model)
    return _dense_model


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        logger.info("Loading sparse embedding model: %s", config.SPARSE_EMBED_MODEL)
        t0 = time.monotonic()
        _sparse_model = SparseTextEmbedding(
            model_name=config.SPARSE_EMBED_MODEL,
            **_onnx_providers_kwargs(),
        )
        logger.info("Sparse embedding model loaded in %.2fs", time.monotonic() - t0)
        _log_actual_provider("Sparse model", _sparse_model)
    return _sparse_model


def preload_models() -> None:
    """
    Force both embedding models to load AND run a real inference pass,
    right now, during server warm-up (synchronously, off the event loop).

    IMPORTANT: constructing a TextEmbedding/SparseTextEmbedding object is
    cheap (~0.3-0.6s, confirmed in the startup log) and is NOT the same
    thing as ONNX Runtime being warm. ONNX Runtime defers a meaningful
    chunk of its real setup cost -- memory arena allocation, kernel
    selection for the actual input shape, spinning up its intra-op thread
    pool for real -- to the FIRST call to .embed(), not to object
    construction. If this function only constructs the models (the old
    behavior), that first-inference cost silently gets paid by whichever
    client upload happens to be first in the session, which is exactly
    the ~15-20s of unexplained slowness seen in testing despite the
    models showing as "loaded" in ~0.6s at startup.

    Fix: run one throwaway batch through _embed_all() here -- the SAME
    concurrent dense+sparse code path a real upload uses (including the
    ONNX thread-cap settings from _onnx_providers_kwargs()) -- so the
    real first-inference cost is paid now, off the request path, instead
    of during the user's first upload.
    """
    _get_dense_model()
    _get_sparse_model()

    t0 = time.monotonic()
    warm_up_batch = [
        "This is a warm-up sentence used only to force ONNX Runtime to "
        "run its first real inference pass now, at server startup, "
        "instead of on the first client upload of the session."
    ] * 4
    _embed_all(warm_up_batch)
    logger.info(
        "Embedding models warmed with a real inference pass in %.2fs "
        "(this is the cost the first real upload used to pay)",
        time.monotonic() - t0,
    )


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            grpc_port=config.QDRANT_GRPC_PORT,
            prefer_grpc=config.QDRANT_PREFER_GRPC,
        )
    return _client


def ensure_collection() -> None:
    """Create the hybrid collection if it doesn't exist yet. Safe to call every startup."""
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if config.QDRANT_COLLECTION in existing:
        return

    dense_dim = len(next(_get_dense_model().embed(["dimension probe"])))

    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={
            "dense": qmodels.VectorParams(size=dense_dim, distance=qmodels.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(),
        },
    )
    logger.info("Created Qdrant collection '%s' (dense dim=%d)", config.QDRANT_COLLECTION, dense_dim)


def extract_text(path: str | Path) -> str:
    """Extract raw text from PDF / DOCX / TXT. Extend here for more types."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # PyMuPDF (fitz) -- noticeably faster than pypdf on longer PDFs.
        import fitz
        doc = fitz.open(str(path))
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()

    if suffix == ".docx":
        import docx
        doc = docx.Document(str(path))
        text_parts = []
        
        # Iterate over all elements in the document body to maintain order
        # For simplicity without complex xml parsing, we'll just extract paragraphs first, then tables.
        # But for better flow, let's extract both:
        for p in doc.paragraphs:
            if p.text.strip():
                text_parts.append(p.text.strip())
                
        for table in doc.tables:
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_data:
                    text_parts.append(" | ".join(row_data))
                    
        return "\n".join(text_parts)

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE_TOKENS * 4,   # rough chars-per-token approximation
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text) if c.strip()]


def reset_collection() -> None:
    """Drop and recreate the collection. Only call this when you actually
    want to wipe previously ingested documents (see reset_first below)."""
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if config.QDRANT_COLLECTION in existing:
        client.delete_collection(config.QDRANT_COLLECTION)
    ensure_collection()


def _embed_all(chunks: list[str]) -> tuple[list, list]:
    """
    Run dense and sparse embedding concurrently instead of sequentially.
    Both are ONNX inference calls that release the GIL for the bulk of
    their work, so two threads genuinely overlap here rather than just
    taking turns. See _onnx_providers_kwargs() / _resolve_onnx_intra_op_threads()
    for the automatic thread-cap heuristic that keeps the two calls from
    contending for CPU cores on constrained hardware.
    """
    with ThreadPoolExecutor(max_workers=2) as ex:
        dense_future = ex.submit(lambda: list(_get_dense_model().embed(chunks)))
        sparse_future = ex.submit(lambda: list(_get_sparse_model().embed(chunks)))
        return dense_future.result(), sparse_future.result()


def ingest_document(
    path: str | Path,
    reset_first: bool = False,
    source_name: str | None = None,
    progress_cb: ProgressCB = None,
    wait_for_durability: bool = False,
) -> int:
    """
    Extract -> chunk -> embed (dense + sparse, concurrently) -> upsert into Qdrant.
    Returns the number of chunks ingested.

    reset_first defaults to False: repeated uploads APPEND to the existing
    collection rather than wiping it. Pass reset_first=True explicitly if
    you want a clean slate (e.g. a "replace all documents" action).

    wait_for_durability defaults to False: the Qdrant upsert returns as
    soon as the write is accepted rather than blocking until it's
    confirmed durable on disk. This is a real, measurable latency win on
    every upload. Pass wait_for_durability=True for call sites where you
    need a hard guarantee the write completed before doing anything else
    (rare for a local dev instance, but available).
    """
    def note(msg: str):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    path = Path(path)
    source_name = source_name or path.name
    t_start = time.monotonic()

    if reset_first:
        note(f"Resetting collection before ingesting '{source_name}'…")
        reset_collection()
    else:
        ensure_collection()

    note(f"Extracting text from '{source_name}'…")
    t0 = time.monotonic()
    text = extract_text(path)
    if not text.strip():
        raise ValueError(f"No extractable text found in {path}")
    logger.info("Text extraction took %.2fs (%d chars)", time.monotonic() - t0, len(text))

    note("Chunking text…")
    t0 = time.monotonic()
    chunks = chunk_text(text)
    logger.info("Chunked '%s' into %d chunks in %.2fs", source_name, len(chunks), time.monotonic() - t0)

    note(f"Embedding {len(chunks)} chunks…")
    t0 = time.monotonic()
    dense_vecs, sparse_vecs = _embed_all(chunks)
    logger.info("Embedding (dense+sparse, concurrent) took %.2fs", time.monotonic() - t0)

    points = []
    for i, (chunk, dvec, svec) in enumerate(zip(chunks, dense_vecs, sparse_vecs)):
        points.append(
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dvec.tolist(),
                    "sparse": qmodels.SparseVector(
                        indices=svec.indices.tolist(),
                        values=svec.values.tolist(),
                    ),
                },
                payload={
                    "text": chunk,
                    "source": source_name,
                    "chunk_index": i,
                },
            )
        )

    note("Storing in Qdrant…")
    t0 = time.monotonic()
    get_qdrant_client().upsert(
        collection_name=config.QDRANT_COLLECTION,
        points=points,
        wait=wait_for_durability,
    )
    logger.info("Qdrant upsert took %.2fs (wait=%s)", time.monotonic() - t0, wait_for_durability)

    logger.info(
        "Ingested %d chunks from '%s' into Qdrant in %.2fs total",
        len(points), source_name, time.monotonic() - t_start,
    )
    return len(points)


def upload_and_ingest(
    path: str | Path,
    source_name: str | None = None,
    reset: bool = False,
    progress_cb: ProgressCB = None,
    wait_for_durability: bool = False,
) -> int:
    """Thin wrapper kept for naming parity with the old pipeline.upload_and_ingest().
    reset=False by default -- see ingest_document()'s reset_first docstring.
    wait_for_durability=False by default -- see ingest_document()'s docstring."""
    return ingest_document(
        path,
        reset_first=reset,
        source_name=source_name,
        progress_cb=progress_cb,
        wait_for_durability=wait_for_durability,
    )