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

# Retrieval fusion: lexical BM25 weights + deterministic token overlap (cheap “second signal”).
HYBRID_BM25_WEIGHT = float(os.environ.get("SUPPORT_AGENT_HYBRID_BM25", "0.64"))
HYBRID_OVERLAP_WEIGHT = float(os.environ.get("SUPPORT_AGENT_HYBRID_OVERLAP", "0.36"))

# LLM
LLM_TEMPERATURE = float(os.environ.get("SUPPORT_AGENT_LLM_TEMPERATURE", "0"))
LLM_MAX_TOKENS = int(os.environ.get("SUPPORT_AGENT_LLM_MAX_TOKENS", "1200"))


def _pick_google_api_key() -> str:
    return (
        os.environ.get("GOOGLE_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
    )


GOOGLE_API_KEY = _pick_google_api_key()
GEMINI_MODEL = os.environ.get("SUPPORT_AGENT_GEMINI_MODEL", "gemini-2.0-flash")
# auto | anthropic | google — auto picks Anthropic when ANTHROPIC_API_KEY is set, else Google when set.
LLM_BACKEND = os.environ.get("SUPPORT_AGENT_LLM_BACKEND", "auto").strip().lower()

# Batch / CLI
BATCH_SLEEP_S = float(os.environ.get("SUPPORT_AGENT_BATCH_SLEEP", "0.05"))

# Embed explainability blob on each triaged ticket (`trace`). Also enabled with CLI `--trace`.
CLI_EMBED_TRACE = os.environ.get("SUPPORT_AGENT_CLI_TRACE", "").strip() in {"1", "true", "yes", "on"}

# Cap retrieval query size for pathological CSV rows (deterministic truncation).
QUERY_MAX_CHARS = int(os.environ.get("SUPPORT_AGENT_QUERY_MAX_CHARS", "32000"))

# Retrieval chunk sizing (overlap stride = max_len - CHUNK_SPLIT_OVERLAP_STRIDE)
CHUNK_BODY_MAX_CHARS = int(os.environ.get("SUPPORT_AGENT_CHUNK_MAX_CHARS", "2000"))
CHUNK_SPLIT_OVERLAP_STRIDE = int(os.environ.get("SUPPORT_AGENT_CHUNK_OVERLAP_STRIDE", "260"))
