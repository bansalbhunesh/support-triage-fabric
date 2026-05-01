"""
Post-generation grounding checks — reduce hallucinated citations and off-corpus URLs.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

ALLOWED_HINT_HOSTS = frozenset(
    {
        "support.hackerrank.com",
        "help.hackerrank.com",
        "support.claude.com",
        "www.visa.co.in",
        "visa.com",
        "usa.visa.com",
        "docs.claude.com",
        "anthropic.com",
        "help.anthropic.com",
        "support.anthropic.com",
        "portal.usepylon.com",
        "claude.ai",
    }
)


def allowlist_urls_from_chunks(chunks: list) -> frozenset[str]:
    urls: set[str] = set()
    for ch in chunks:
        u = getattr(ch, "source_url", None) or ""
        u = u.strip().rstrip("/")
        if u:
            urls.add(u)
    return frozenset(urls)


_URL_RE = re.compile(r"https?://[^\s\]>\),]+", re.I)
_TEL_RE = re.compile(r"\btel:\+?[\d\s\-().]{6,45}\b", re.I)
_MAILTO_RE = re.compile(r"mailto:[^\s\]>\),]+", re.I)


def extract_urls_from_text(text: str) -> list[str]:
    """Pull http(s) URLs from model output for conservative checks."""
    if not text:
        return []
    found: list[str] = []
    for m in _URL_RE.findall(text):
        found.append(m.rstrip(".,);\"')"))
    return found


def extract_tel_links(text: str) -> list[str]:
    if not text:
        return []
    return [m.rstrip(".,);\"')") for m in _TEL_RE.findall(text)]


def extract_mailto_links(text: str) -> list[str]:
    if not text:
        return []
    return [m.rstrip(".,);\"')") for m in _MAILTO_RE.findall(text)]


def _tel_digits_ok(raw: str) -> bool:
    u = (raw or "").strip()
    if not u.lower().startswith("tel:"):
        return False
    digits = re.sub(r"\D", "", u[4:])
    return 8 <= len(digits) <= 18


def _mailto_support_ok(raw: str) -> bool:
    low = (raw or "").lower()
    for host in (
        "hackerrank.com",
        "anthropic.com",
        "visa.com",
        "claude.com",
        "usepylon.com",
    ):
        if host in low:
            return True
    return False


def _same_or_prefix(u: str, allowlist: frozenset[str]) -> bool:
    u = u.rstrip("/")
    if u in allowlist:
        return True
    for a in allowlist:
        if not a:
            continue
        if u.startswith(a.rstrip("/") + "/") or u == a.rstrip("/"):
            return True
    return False


def _host_trusted(hostname: str) -> bool:
    hl = (hostname or "").lower()
    if not hl:
        return False
    return any(hl == th or hl.endswith("." + th) or hl.endswith(th) for th in ALLOWED_HINT_HOSTS)


def grounding_violations(
    cited_sources: list[str],
    allowlist: frozenset[str],
    response_text: str = "",
    check_body_urls: bool = False,
) -> tuple[bool, str]:
    """
    Fail closed when cited_urls are not subsets of retrieval allowlist.
    Optional scan of response URLs (stricter — may disagree with models that cite Stripe,
    Gmail, etc.); default off so we prioritize citationsubset integrity first.
    """
    cites = [(c or "").strip().rstrip("/") for c in cited_sources if (c or "").strip()]
    for u in cites:
        if not _same_or_prefix(u, allowlist):
            return False, f"citation_not_in_evidence:{u[:100]}"

    if check_body_urls and response_text:
        for u in extract_urls_from_text(response_text):
            host = urlparse(u).hostname or ""
            if _host_trusted(host) or _same_or_prefix(u.rstrip("/"), allowlist):
                continue
            return False, f"unsupported_url_in_body:{host}"
        for t in extract_tel_links(response_text):
            if _tel_digits_ok(t):
                continue
            return False, f"unsupported_tel_in_body:{t[:80]}"
        for m in extract_mailto_links(response_text):
            if _mailto_support_ok(m):
                continue
            return False, f"unsupported_mailto_in_body:{m[:100]}"

    return True, "ok"


def evidence_allowlist_extended(allowlist: frozenset[str], *extra: str) -> frozenset[str]:
    s = set(allowlist)
    for e in extra:
        e = (e or "").strip().rstrip("/")
        if e:
            s.add(e)
    return frozenset(s)
