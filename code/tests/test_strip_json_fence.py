"""Tests for JSON fence stripping (stdlib unittest — no pytest required)."""

from __future__ import annotations

import pathlib
import sys
import unittest

_CODE = pathlib.Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from models import strip_json_fence  # noqa: E402 — path bootstrap above


class TestStripJsonFence(unittest.TestCase):
    def test_plain_json(self):
        raw = '{"status":"replied","sources":[]}'
        self.assertEqual(strip_json_fence(raw), raw)

    def test_triple_fence_with_json_label(self):
        raw = "```json\n{\"request_type\":\"bug\"}\n```"
        self.assertEqual(strip_json_fence(raw), '{"request_type":"bug"}')

    def test_nested_or_double_fence_fallback(self):
        raw = "```\n```json\n{\"a\":1}\n```\n```"
        out = strip_json_fence(raw)
        self.assertIn('"a"', out)
        self.assertEqual(out.strip().startswith("{"), True)

    def test_trailing_backticks(self):
        raw = "```\n{\"x\":true}\n```\n`\n"
        self.assertEqual(strip_json_fence(raw), '{"x":true}')

    def test_empty(self):
        self.assertEqual(strip_json_fence(""), "")
        self.assertEqual(strip_json_fence(None), "")


if __name__ == "__main__":
    unittest.main()
