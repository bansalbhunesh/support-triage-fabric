"""
Unified Anthropic Claude vs Google Gemini client for corpus-grounded triage synthesis.
Reads keys only from the environment — never bundle secrets.

The hackathon does **not** require an Anthropic key: set `GOOGLE_API_KEY` (or
`GEMINI_API_KEY`) and optionally `SUPPORT_AGENT_LLM_BACKEND=google`. The
`anthropic` package is only loaded when that backend is selected (see
`requirements-gemini.txt` for a minimal install without it).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Literal

from config import (
    GEMINI_MODEL,
    HTTP_TIMEOUT_S,
    LLM_BACKEND,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MODEL,
)

LogFn = Callable[[str, str], None]


def _google_key_live() -> str:
    return os.environ.get("GOOGLE_API_KEY", "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()


@dataclass(frozen=True)
class AgentLlm:
    """Exactly one synthesis backend is active."""

    backend: Literal["anthropic", "google"]
    model_name: str
    anthropic_client: Any = None  # anthropic.Anthropic when backend is anthropic
    gemini_api_key: str = ""

    @property
    def label(self) -> str:
        return f"{self.backend}:{self.model_name}"


def _anthropic_sdk():
    """Import Anthropic SDK only when that backend is used (Gemini-only installs skip it)."""
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "Anthropic synthesis backend requested but SDK not installed: pip install anthropic "
            "(Gemini-only: set SUPPORT_AGENT_LLM_BACKEND=google with GOOGLE_API_KEY or GEMINI_API_KEY; "
            "leave ANTHROPIC_API_KEY unset and use pip install -r code/requirements-gemini.txt)."
        ) from e
    return anthropic


def build_agent_llm() -> AgentLlm | None:
    anth_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    g_key = _google_key_live()

    bk_raw = LLM_BACKEND
    if bk_raw == "anthropic":
        if not anth_key:
            raise RuntimeError(
                "SUPPORT_AGENT_LLM_BACKEND=anthropic requires ANTHROPIC_API_KEY in the environment"
            )
        sdk = _anthropic_sdk()
        return AgentLlm(
            "anthropic",
            MODEL,
            anthropic_client=sdk.Anthropic(api_key=anth_key, timeout=HTTP_TIMEOUT_S),
            gemini_api_key="",
        )
    if bk_raw == "google":
        if not g_key:
            raise RuntimeError(
                "SUPPORT_AGENT_LLM_BACKEND=google requires GOOGLE_API_KEY or GEMINI_API_KEY"
            )
        return AgentLlm("google", GEMINI_MODEL, anthropic_client=None, gemini_api_key=g_key)

    # auto
    if anth_key:
        sdk = _anthropic_sdk()
        return AgentLlm(
            "anthropic",
            MODEL,
            anthropic_client=sdk.Anthropic(api_key=anth_key, timeout=HTTP_TIMEOUT_S),
            gemini_api_key="",
        )
    if g_key:
        return AgentLlm("google", GEMINI_MODEL, anthropic_client=None, gemini_api_key=g_key)
    return None


def synthesize_json_turn(agent: AgentLlm, system: str, user: str, log: LogFn) -> str:
    """Return raw model output (prefer JSON-only)."""
    if agent.backend == "anthropic":
        assert agent.anthropic_client is not None
        msg = agent.anthropic_client.messages.create(
            model=agent.model_name,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        blocks = getattr(msg, "content", None) or []
        for blk in blocks:
            t = getattr(blk, "text", None)
            if isinstance(t, str) and t.strip():
                return t.strip()
        raise RuntimeError("Anthropic returned no textual content blocks.")

    return _synthesize_gemini_json(agent.gemini_api_key, agent.model_name, system, user, log)


def _synthesize_gemini_json(
    api_key: str,
    model_id: str,
    system: str,
    user: str,
    log: LogFn,
) -> str:
    try:
        import google.generativeai as genai  # lazy: optional dependency
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Google backend requires `pip install google-generativeai` (see code/requirements.txt)."
        ) from e

    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(
        model_id,
        system_instruction=system.strip()[:31_900] if system.strip() else None,
    )

    cfg = genai.types.GenerationConfig(
        candidate_count=1,
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_TOKENS,
        response_mime_type="application/json",
    )

    req_opts = {"timeout": HTTP_TIMEOUT_S}
    try:
        from google.generativeai.types import RequestOptions

        req_opts = RequestOptions(timeout=HTTP_TIMEOUT_S)
    except ImportError:
        pass

    resp = gm.generate_content(user, generation_config=cfg, request_options=req_opts)
    if not resp.candidates:
        fb = getattr(resp, "prompt_feedback", None)
        log("GEMINI_BLOCK", repr(fb) if fb else "no_candidates")
        raise RuntimeError("Gemini returned no candidates (blocked or filtered).")

    part = getattr(resp, "text", None)
    if isinstance(part, str) and part.strip():
        return part.strip()

    if resp.candidates[0].content.parts:
        t = "".join(getattr(p, "text", "") for p in resp.candidates[0].content.parts)
        return t.strip()

    log("GEMINI_EMPTY", repr(resp.prompt_feedback))
    raise RuntimeError("Gemini candidate had no textual content.")
