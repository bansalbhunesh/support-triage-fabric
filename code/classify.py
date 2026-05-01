"""
Classification helpers: domain detection with confidence, enum request_type,
product_area derived from evidence.
"""

from __future__ import annotations

import re
from typing import Iterable

from retriever import Chunk


DOMAIN_KW: dict[str, list[str]] = {
    "hackerrank": [
        "hackerrank",
        "hacker rank",
        "assessment",
        "coding test",
        "screen",
        "candidate",
        "recruiter",
        "proctoring",
        "interview",
        "mock interview",
        "library",
        "chakra",
        "engage",
        "submission",
        "challenge",
        "coding challenge",
    ],
    "claude": [
        "claude",
        "anthropic",
        "claude.ai",
        "opus",
        "bedrock",
        "vertex",
        "lti",
        "sso",
        "workspace",
        "subscription",
    ],
    "visa": [
        "visa",
        "chargeback",
        "dispute",
        "card declined",
        "merchant minimum",
        "minimum",
        "spend requirement",
        "block",
        "blocké",
        "bloqué",
        "bloquee",
        "Issuer",
        "issuing bank",
        "lost card",
        "stolen",
    ],
}


def normalize_company(company_raw: str | None) -> str | None:
    if not company_raw:
        return None
    c = company_raw.strip().lower()
    if c in ("none", "n/a", "unknown", ""):
        return None
    if "hackerrank" in c:
        return "hackerrank"
    if "claude" in c:
        return "claude"
    if "visa" in c:
        return "visa"
    return None


def detect_domain_scores(text: str) -> tuple[dict[str, int], str | None]:
    tl = text.lower()
    scores = {d: sum(1 for kw in kws if kw in tl) for d, kws in DOMAIN_KW.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_d, best_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else 0
    if best_s == 0:
        return scores, None
    ambiguous = best_s > 0 and (best_s - second_s) <= 2 and second_s > 0
    note = "ambiguous_top_domains" if ambiguous else None
    return scores, note


def infer_domain(company: str | None, issue: str, subject: str) -> tuple[str, dict[str, int], str | None]:
    blob = f"{issue}\n{subject}"
    normalized = normalize_company(company)
    if normalized:
        return normalized, {normalized: 999}, None
    scores, amb = detect_domain_scores(blob)
    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return "unknown", scores, amb or "unknown_domain"
    return best, scores, amb


def classify_request_type(issue: str, subject: str) -> str:
    """Map to hackathon enum: product_issue | feature_request | bug | invalid."""
    raw = (issue + "\n" + subject).strip()
    blob = raw.lower()

    hostile = re.search(
        r"(delete all files|rm -rf|format the disk|give me the code to wipe|wipe the server)", blob
    )
    if hostile:
        return "invalid"

    if len(blob.split()) <= 8 and blob.strip().rstrip(",.!😀😊👍").lower() in (
        "thanks",
        "thank you",
        "thx",
        "thanks!",
        "ty",
        "ok",
        "okay",
        "sure",
        "great",
        "received",
        "got it",
        "thank you,",
        "thanks,",
    ):
        return "invalid"

    if re.search(
        r"\b(who\s+(is|was|played))\b.+"
        r"\b(actor|celebrity|movie|oscar)\b|\b(movie plot|celebrity gossip)\b",
        blob,
    ):
        return "invalid"

    if re.search(
        r"\b(request|please add|can you add|we need the ability|would like to see)\b.*\b(feature|integration|support for)\b",
        blob,
    ) or re.search(
        r"\b(feature request|new capability|enhancement request|extend inactivity|longer timeout|bit more time)\b",
        blob,
    ) or re.search(
        r"\b(extend|increase|adjust|lengthen)\b[\s\S]{0,48}\b(inactivity|session|idle)\b[\s\S]{0,32}\b"
        r"(timeout|time[- ]?out)\b",
        blob,
    ) or re.search(
        r"\b(can we|please|would like to)\b[\s\S]{0,32}\b(extend|increase)\b[\s\S]{0,40}\bsession\b[\s\S]{0,36}\b"
        r"(minutes?|hours?)\b",
        blob,
    ) or re.search(r"\bpause\b[\s\S]{0,52}\bsubscription\b|\bsubscription\b[\s\S]{0,52}\bpause\b", blob):
        return "feature_request"

    bug_signals = [
        "not working",
        "doesn't work",
        "does not work",
        "stopped working",
        "stopped in between",
        "error",
        "bug",
        "is down",
        "are down",
        "unable to",
        "can't ",
        "cannot ",
        "blocked",
        "failing",
        "connectivity",
        "compatibility",
        "no submissions",
        "not able to see",
        "isn't working",
    ]
    if any(s in blob for s in bug_signals):
        return "bug"

    return "product_issue"


def infer_product_area(domain: str, top_chunks: Iterable[Chunk]) -> str:
    breadcrumb_guess = ""
    chunks = list(top_chunks)
    if chunks:
        ch0 = chunks[0]
        breadcrumb_guess = ch0.product_hint.replace("/", " · ")[:120]
    defaults = {
        "hackerrank": "HackerRank Platform",
        "claude": "Claude · Anthropic Help Center",
        "visa": "Visa Consumer Support",
        "unknown": "Cross-domain / Unknown",
    }
    base = defaults.get(domain, defaults["unknown"])
    if breadcrumb_guess:
        return f"{base} — {breadcrumb_guess}"
    return base


_NON_ACCESS_USES = re.compile(
    r"\b(test link|assessment link|public link|invite link|invite url|candidate invite|"
    r"invite page|broken invite|public test link|error when accessing|cannot access the test|"
    r"accessing the test|link is not working)\b",
    re.I,
)


def legacy_request_label(issue: str, subject: str, domain: str) -> str:
    """
    Human-readable routing label (logging / auxiliary),
    ordered to avoid swallowing nuanced cases behind generic 'access'.
    """
    tl = (issue + "\n" + subject).lower()

    if domain in {"visa", "unknown"} and re.search(
        r"\b(card|carte)\b.{0,32}\b(stolen|vol[eé]|lost|lost my|picked|pickpocket|misplaced)|"
        r"\b(stolen|lost)\b.{0,24}\b(card|carte)|\bmisplaced\b.{0,20}\bcard\b|"
        r"\bcarte\b.{0,36}\b(perdu|vol[eé])\b|\bpick[- ]pocket\b",
        tl,
        re.I,
    ):
        return "Lost or Stolen Card"

    link_or_test_issue = bool(
        _NON_ACCESS_USES.search(tl)
        or ("link" in tl and ("error" in tl or "not working" in tl))
        or ("assessment link" in tl)
        or ("invite link" in tl)
        or ("cannot access the test" in tl)
        or ("accessing the test" in tl and "error" in tl)
        or ("submission" in tl and "working" in tl)
        or ("submissions" in tl and ("not working" in tl or "none of" in tl))
    )

    if link_or_test_issue:
        return "Assessment / Access or Link Troubleshooting"

    if domain == "hackerrank" and any(w in tl for w in ("password", "forgot password")):
        # Avoid tagging assessment / invite link breakage as generic password-reset triage.
        if not link_or_test_issue:
            return "Account Access / Password Reset"
    if (
        domain == "hackerrank"
        and ("log in" in tl or "login" in tl or "can't access my account" in tl or "cannot access my account" in tl)
        and not link_or_test_issue
        and "workspace" not in tl
    ):
        return "Account Access / Password Reset"

    if any(w in tl for w in ("billing", "charge", "invoice", "subscription", "refund", "payment")):
        return "Billing / Payment"
    if any(w in tl for w in ("integration", "ats", "sso", "okta", "greenhouse")):
        return "Integration / Technical Setup"
    if "library" in tl or "question library" in tl:
        return "Question Library"
    if any(w in tl for w in ("invite candidates", "assessment", "coding test")) and not link_or_test_issue:
        return "Assessment / Test Management"
    if "execution environment" in tl or ("python" in tl and "version" in tl):
        return "Execution Environment"
    if "visa" in tl and "minimum" in tl:
        return "Merchant Rules"

    return "General Inquiry"


def format_evidence_urls(chunks: list[Chunk], max_n: int = 3) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for ch in chunks:
        u = (ch.source_url or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= max_n:
            break
    return out
