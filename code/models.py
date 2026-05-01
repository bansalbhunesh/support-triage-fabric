"""Structured outputs from the LLM path (validated with Pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LlmStructuredReply(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    status: Literal["replied", "escalated"]
    product_area: str = Field(..., max_length=600)
    response: str = Field(..., max_length=20000)
    justification: str = Field(..., max_length=6000)
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]
    sources: list[str] = Field(default_factory=list, max_length=12)


def strip_json_fence(text: str) -> str:
    """Extract JSON from model output; tolerate multiple fences or stray prose."""
    raw = (text or "").strip()
    if not raw:
        return raw
    if raw.startswith("```"):
        parts = raw.split("```")
        chunk = ""
        if len(parts) >= 2:
            chunk = parts[1].lstrip("\n").lstrip()
        if len(chunk) < 8 and len(parts) >= 4:
            chunk = parts[2].lstrip("\n").lstrip()
        raw = chunk or raw
    if raw.lower().startswith("json"):
        raw = raw[4:].lstrip()
    return raw.strip().strip("`").strip()
