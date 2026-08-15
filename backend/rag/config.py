"""
Central config for backend/rag/. Change model/collection choices here only —
nothing else in this package should hardcode these values.
"""
import os

# --- Qdrant (server mode) ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "a2f_documents")
QDRANT_PREFER_GRPC = False  # faster than REST; falls back gracefully if unavailable
DENSE_EMBED_MODEL = os.getenv("DENSE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SPARSE_EMBED_MODEL = os.getenv("SPARSE_EMBED_MODEL", "Qdrant/bm25")

# --- Chunking ---
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "250"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "25"))

# --- Retrieval ---
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "5"))
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"  # off by default for v1

# --- Groq (STT + generation) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # required — set in .env, never commit
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_LLM_TEMPERATURE = float(os.getenv("GROQ_LLM_TEMPERATURE", "0.3"))

if not GROQ_API_KEY:
    import warnings
    warnings.warn(
        "GROQ_API_KEY is not set. RAG generation and STT will fail at call time. "
        "Set it in your environment or .env file."
    )
ONNX_INTRA_OP_THREADS = 0
USE_CUDA_EMBEDDING = False