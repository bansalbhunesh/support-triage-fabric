"""
Risk analysis, abuse/prompt-injection heuristics, and escalation tiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from corpus import ESCALATION_RULES, SUPPORT_CORPUS


@dataclass(frozen=True)
class RiskSignal:
    escalate: bool
    tier: str  # "none" | "soft" | "hard"
    reason: str
    block: str

_INJECTION_PATTERNS = [
    r"ignore previous",
    r"jailbreak",
    r"internal rules",
    r"règles internes",
    r"logic exact",
    r"system prompt",
    r"reveal your",
    r"full policy",
    r"exact.*rules",
    r"documents récupérés",
]

_MALICIOUS_ASKS = [
    r"delete all files",
    r"format\s+the\s+disk",
    r"rm\s+-rf",
]

_SCORE_DISPUTE = re.compile(
    r"(increase my score|graded me unfair|review my answers|move me to the next round)",
    re.I,
)
_VISA_REFUND_DEMAND = re.compile(
    r"(refund me|ban the seller|visa refund|charge back today)", re.I
)
_CERTIFICATE_FIX = re.compile(r"(name is incorrect on the certificate|update my name on the certificate)", re.I)
_ACCOUNT_TAKEOVER_ADMIN = re.compile(
    r"(not the workspace owner|not the workspace admin|not the admin).*(restore|lost).*(access|seat)|"
    r"(restore|regain).*(access|seat).*(not the workspace owner|not the workspace admin)",
    re.I | re.DOTALL,
)


def _visa_stolen_or_fraud(text: str) -> bool:
    tl = text.lower()
    if "visa" in tl and any(
        x in tl for x in ("stolen", "lost my card", "lost card", "fraud", "identity theft", "identity stolen")
    ):
        return True
    if re.search(r"\bcard was stolen\b|\bstolen at\b|\blost/stolen\b", tl):
        return True
    return False


def heuristic_risk_scan(issue_text: str, subject: str, domain: str) -> RiskSignal:
    blob = f"{issue_text}\n{subject}".strip()
    tl = blob.lower()

    for pat in _INJECTION_PATTERNS:
        if re.search(pat, blob, re.I):
            return RiskSignal(
                True,
                "hard",
                "prompt_manipulation_or_sensitive_disclosure_request",
                "",
            )

    for pat in _MALICIOUS_ASKS:
        if re.search(pat, tl):
            return RiskSignal(True, "hard", "potentially_harmful_or_out_of_scope_request", "")

    if _SCORE_DISPUTE.search(blob):
        return RiskSignal(True, "hard", "test_outcome_or_score_dispute_requires_human_review", "")

    if domain == "visa" and _VISA_REFUND_DEMAND.search(blob):
        return RiskSignal(True, "soft", "merchant_refund_or_seller_action_requires_issuer", "")

    if _CERTIFICATE_FIX.search(blob):
        return RiskSignal(True, "soft", "credential_or_certificate_correction_requires_support", "")

    if _ACCOUNT_TAKEOVER_ADMIN.search(blob):
        return RiskSignal(True, "soft", "seat_or_admin_change_requires_org_admin_or_support", "")

    if re.search(r"\bsecurity vulnerability\b|\bbug bounty\b", tl) and domain == "claude":
        return RiskSignal(True, "hard", "security_disclosure_requires_specialist_channel", "")

    if re.search(r"\bidentity (has been )?stolen\b|\bidentity theft\b", tl) and domain == "visa":
        return RiskSignal(True, "hard", "identity_theft_or_high_risk_payment_event", "")

    if domain == "hackerrank":
        if any(
            k in tl for k in ("infosec", "information security", "vendor security", "security questionnaire", "soc 2", "soc2")
        ) and any(
            k in tl
            for k in (
                "form",
                "forms",
                "filling",
                "questionnaire",
                "assessment package",
                "compliance",
                "audit",
                "vendor",
            )
        ):
            return RiskSignal(True, "soft", "enterprise_security_review_requires_account_team", "")

        if (
            (
                "none of" in tl
                and "are working" in tl
                and any(k in tl for k in ("submission", "submissions", "challenge", "challenges"))
            )
            or (
                ("whole site" in tl or "entire platform" in tl or "everything is" in tl)
                and any(
                    k in tl
                    for k in (
                        "not working",
                        "aren't working",
                        "down",
                        "broken",
                        "offline",
                    )
                )
            )
            or "site is down" in tl
            or re.search(r"none of the pages", tl)
        ):
            return RiskSignal(True, "hard", "widespread_platform_issue_requires_operations_team", "")

    if domain == "visa" and re.search(r"\burgent\b.*\bcash\b|\bcash advance\b|\bneed cash\b", tl):
        return RiskSignal(True, "soft", "cash_access_or_credit_line_requires_issuer", "")

    return RiskSignal(False, "none", "", "")


def corpus_escalation(issue_text: str, domain: str) -> tuple[bool, str]:
    if domain not in ESCALATION_RULES or domain == "unknown":
        return False, ""

    tl = issue_text.lower()

    if domain == "visa" and _visa_stolen_or_fraud(issue_text):
        return True, "lost_stolen_fraud_or_identity_event"

    rules = ESCALATION_RULES.get(domain, {})
    for phrase in rules.get("always_escalate", []):
        if phrase.lower() in tl:
            return True, phrase

    triggers = SUPPORT_CORPUS.get(domain, {}).get("escalate_triggers", [])
    urgent = any(
        w in tl
        for w in ("urgent", "immediately", "asap", "emergency", "unauthorized charge", "unauthorized", "hacked")
    )
    for t in triggers:
        if t.lower() in tl and urgent:
            return True, t

    if domain == "hackerrank" and ("hacked" in tl or "account compromise" in tl) and (
        "email" in tl or "recruiter" in tl or "changed" in tl
    ):
        return True, "account_compromise_claim"

    if domain == "claude" and "gdpr" in tl and any(
        x in tl for x in ("delete", "deletion", "erase", "remove all my data")
    ):
        return True, "gdpr_or_data_deletion_requires_privacy_team"

    return False, ""


def escalation_message_for(domain: str) -> str:
    return ESCALATION_RULES.get(domain, {}).get(
        "escalation_message",
        "Please contact the appropriate support team using the official help center.",
    )
