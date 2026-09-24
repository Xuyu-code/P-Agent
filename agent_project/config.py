"""Central configuration for the Hehuang heritage QA + creation agent.

All values are read from environment variables (see .env.example). The agent
is a pure CPU service: the LLM and (by default) embeddings live behind remote
OpenAI-compatible APIs; the GPU is reserved for the shadow-puppet generation
service, which this project only talks to over HTTP.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AGENT_ROOT = Path(__file__).resolve().parents[1]
DATA_SOURCES_DIR = AGENT_ROOT / "data_sources"
HERITAGE_CARDS_DIR = DATA_SOURCES_DIR / "heritage_public_facts"
PROJECT_FACTS_DIR = DATA_SOURCES_DIR / "project_facts"

# Runtime data: Chroma index + user-uploaded lineart. Never committed to git.
DATA_DIR = AGENT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
UPLOADS_DIR = DATA_DIR / "uploads"


def ensure_data_dirs() -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# External image-generation service — HTTP only
# ---------------------------------------------------------------------------
PUPPET_API_BASE = os.environ.get("PUPPET_API_BASE", "http://localhost:8000").rstrip("/")
PUPPET_SUBMIT_TIMEOUT = float(os.environ.get("PUPPET_SUBMIT_TIMEOUT", "30"))
PUPPET_POLL_INTERVAL = float(os.environ.get("PUPPET_POLL_INTERVAL", "2"))
PUPPET_POLL_TIMEOUT = float(os.environ.get("PUPPET_POLL_TIMEOUT", "300"))

# Default controls for the external generation service.
DEFAULT_K = 3
DEFAULT_STEPS = 40
DEFAULT_GUIDANCE_SCALE = 8.0
DEFAULT_CONDITIONING_SCALE = 0.5
DEFAULT_SEED = 42
DEFAULT_POSTPROCESS = True

# ---------------------------------------------------------------------------
# LLM (OpenAI-compatible API: DashScope / Kimi / ...)
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-plus")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "180"))  # 推理模型+大证据 prompt 单次可超 60s
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# Embeddings: "dashscope" (remote, default) or "local" (BGE via hf-mirror)
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "dashscope").strip().lower()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "1024"))  # DashScope text-embedding-v4
# Local provider: BAAI/bge-small-zh-v1.5 downloaded once via hf-mirror, then cached.
LOCAL_EMBEDDING_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
LOCAL_EMBEDDING_DIM = 512

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
COLLECTION_HERITAGE = "heritage_facts"
COLLECTION_PROJECT = "project_facts"
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "4"))
# Cosine distance threshold: hits farther than this are treated as "no evidence".
RETRIEVAL_MAX_DISTANCE = float(os.environ.get("RETRIEVAL_MAX_DISTANCE", "0.55"))
# The project-service collection uses a separate threshold because its
# documents have a different length and vocabulary distribution.
RETRIEVAL_MAX_DISTANCE_PROJECT = float(os.environ.get("RETRIEVAL_MAX_DISTANCE_PROJECT", "0.62"))
# Chunking for long project documents (heritage cards stay one-doc-per-card).
PROJECT_CHUNK_SIZE = 800
PROJECT_CHUNK_OVERLAP = 120
