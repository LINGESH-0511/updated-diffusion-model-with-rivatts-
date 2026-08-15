"""
Application Configuration
Loads environment variables from .env
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ==========================
# Audio2Face Configuration
# ==========================

A2F_GRPC_HOST = os.getenv("A2F_GRPC_HOST", "127.0.0.1")
A2F_GRPC_PORT = int(os.getenv("A2F_GRPC_PORT", "52000"))

A2F_HTTP_HOST = os.getenv("A2F_HTTP_HOST", "127.0.0.1")
A2F_HTTP_PORT = int(os.getenv("A2F_HTTP_PORT", "8000"))

# ==========================
# Logging
# ==========================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ==========================
# API Keys
# ==========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NGC_API_KEY = os.getenv("NGC_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
# ==========================
# URLs
# ==========================

HTTP_BASE_URL = f"http://{A2F_HTTP_HOST}:{A2F_HTTP_PORT}"
GRPC_TARGET = f"{A2F_GRPC_HOST}:{A2F_GRPC_PORT}"