"""
Central configuration (env-backed). Imported by pipeline modules only — no corpus imports here.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL = os.environ.get("SUPPORT_AGENT_MODEL", "claude-sonnet-4-6")
DATA_DIR = Path(os.environ.get("SUPPORT_CORPUS_ROOT", str(REPO_ROOT / "data")))
DEFAULT_LOG = REPO_ROOT / "logs" / "log.txt"
LOG_FILE = Path(os.environ.get("SUPPORT_AGENT_LOG", str(DEFAULT_LOG)))
CACHE_DIR = Path(os.environ.get("SUPPORT_AGENT_CACHE_DIR", str(REPO_ROOT / "logs")))

# Retrieval fusion: BM25 + token overlap + deterministic semantic projection (no extra model deps).
HYBRID_BM25_WEIGHT = float(os.environ.get("SUPPORT_AGENT_HYBRID_BM25", "0.52"))
HYBRID_OVERLAP_WEIGHT = float(os.environ.get("SUPPORT_AGENT_HYBRID_OVERLAP", "0.30"))
HYBRID_SEMANTIC_WEIGHT = float(os.environ.get("SUPPORT_AGENT_HYBRID_SEMANTIC", "0.18"))
SEMANTIC_HASH_DIM = int(os.environ.get("SUPPORT_AGENT_SEMANTIC_DIM", "256"))
SEMANTIC_HASH_BUCKETS = int(os.environ.get("SUPPORT_AGENT_SEMANTIC_BUCKETS", "4096"))

# Optional dense embeddings (sentence-transformers). See code/requirements-embeddings.txt
EMBEDDING_BACKEND = os.environ.get("SUPPORT_AGENT_EMBEDDING_BACKEND", "none").strip().lower()
EMBEDDING_MODEL = os.environ.get("SUPPORT_AGENT_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
EMBEDDING_BATCH = int(os.environ.get("SUPPORT_AGENT_EMBEDDING_BATCH", "32"))
HYBRID_DENSE_WEIGHT = float(os.environ.get("SUPPORT_AGENT_HYBRID_DENSE", "0.14"))
DENSE_INPUT_MAX_CHARS = int(os.environ.get("SUPPORT_AGENT_DENSE_INPUT_MAX_CHARS", "2000"))

# LLM
LLM_TEMPERATURE = float(os.environ.get("SUPPORT_AGENT_LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.environ.get("SUPPORT_AGENT_LLM_MAX_TOKENS", "1200"))
# Cap subject+issue text in the LLM user message (tokens + provider limits).
LLM_USER_BLOB_MAX_CHARS = int(os.environ.get("SUPPORT_AGENT_LLM_USER_MAX_CHARS", "26000"))


def _pick_google_api_key() -> str:
    return (
        os.environ.get("GOOGLE_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
    )


GOOGLE_API_KEY = _pick_google_api_key()
GEMINI_MODEL = os.environ.get("SUPPORT_AGENT_GEMINI_MODEL", "gemini-2.0-flash")
# auto | anthropic | google — auto picks Anthropic when ANTHROPIC_API_KEY is set, else Google when set.
LLM_BACKEND = os.environ.get("SUPPORT_AGENT_LLM_BACKEND", "auto").strip().lower()

# HTTP timeouts for synthesis (Anthropic SDK / shared httpx semantics)
HTTP_TIMEOUT_S = float(os.environ.get("SUPPORT_AGENT_HTTP_TIMEOUT_S", "120"))

# Batch / CLI
BATCH_SLEEP_S = float(os.environ.get("SUPPORT_AGENT_BATCH_SLEEP", "0.05"))

# Embed explainability blob on each triaged ticket (`trace`). Also enabled with CLI `--trace`.
CLI_EMBED_TRACE = os.environ.get("SUPPORT_AGENT_CLI_TRACE", "").strip() in {"1", "true", "yes", "on"}

# Cap retrieval query size for pathological CSV rows (deterministic truncation).
QUERY_MAX_CHARS = int(os.environ.get("SUPPORT_AGENT_QUERY_MAX_CHARS", "32000"))

# Retrieval chunk sizing (overlap stride = max_len - CHUNK_SPLIT_OVERLAP_STRIDE)
CHUNK_BODY_MAX_CHARS = int(os.environ.get("SUPPORT_AGENT_CHUNK_MAX_CHARS", "2000"))
CHUNK_SPLIT_OVERLAP_STRIDE = int(os.environ.get("SUPPORT_AGENT_CHUNK_OVERLAP_STRIDE", "260"))

# Domain-conditioned score multiplier in retriever.search (confirmed company ⇒ stronger bias)
DEFAULT_DOMAIN_HINT_BOOST = float(os.environ.get("SUPPORT_AGENT_DOMAIN_HINT_BOOST", "1.38"))
DEFAULT_DOMAIN_CONFIRMED_BOOST = float(os.environ.get("SUPPORT_AGENT_DOMAIN_CONFIRMED_BOOST", "3.1"))

# When true, scan reply body URLs against allowlist / tel: / issuer-safe hosts after citations validate.
GROUNDING_SCAN_RESPONSE_URLS = os.environ.get("SUPPORT_AGENT_GROUND_BODY_URLS", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
