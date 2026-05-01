"""Heuristic checks for classify + routing labels."""

from __future__ import annotations

import pathlib
import sys
import unittest

_CODE = pathlib.Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from classify import classify_request_type, legacy_request_label  # noqa: E402


class TestClassifyPatterns(unittest.TestCase):
    def test_stolen_card_label_without_visa_company(self):
        txt = "my card was stolen last night outside the subway"
        self.assertEqual(legacy_request_label(txt, "", "unknown"), "Lost or Stolen Card")

    def test_feature_extend_inactivity(self):
        body = (
            "We use Claude for onboarding; can we extend the inactivity session timeout by a "
            "few more minutes?"
        )
        self.assertEqual(classify_request_type(body, ""), "feature_request")

    def test_pause_subscription_feature(self):
        self.assertEqual(classify_request_type("", "pause my team subscription").lower(), "feature_request")

    def test_invalid_thanks(self):
        self.assertEqual(classify_request_type("Thank you,", ""), "invalid")

    def test_invite_before_password_when_link_issue(self):
        tl = (
            "The candidate invite URL throws a browser error immediately on open; password reset docs don't help.\n\n"
            "Please advise."
        )
        self.assertEqual(
            legacy_request_label(tl, "", "hackerrank"),
            "Assessment / Access or Link Troubleshooting",
        )


if __name__ == "__main__":
    unittest.main()
