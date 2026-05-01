"""Grounding body URL / tel / mailto relaxations."""

from __future__ import annotations

import pathlib
import sys
import unittest

_CODE = pathlib.Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from grounding import grounding_violations  # noqa: E402


class TestGroundingBody(unittest.TestCase):
    def test_tel_allowed_in_body_scan(self):
        allow = frozenset({"https://support.hackerrank.com/articles/1-test"})
        cites = ["https://support.hackerrank.com/articles/1-test"]
        body = "Call us at tel:+1-800-555-0199 for urgent help."
        ok, _ = grounding_violations(cites, allow, body, check_body_urls=True)
        self.assertTrue(ok)

    def test_random_http_still_blocked(self):
        allow = frozenset({"https://support.hackerrank.com/articles/1-test"})
        cites = ["https://support.hackerrank.com/articles/1-test"]
        body = "See https://evil.example/phish for details."
        ok, reason = grounding_violations(cites, allow, body, check_body_urls=True)
        self.assertFalse(ok)
        self.assertIn("unsupported_url", reason)


if __name__ == "__main__":
    unittest.main()
