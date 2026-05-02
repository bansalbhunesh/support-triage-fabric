"""Defensive normalization for official eval CSV columns."""

from __future__ import annotations

import pathlib
import sys
import unittest

_CODE = pathlib.Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))


class TestOfficialCsvSanitize(unittest.TestCase):
    def test_unknown_status_becomes_escalated(self):
        import agent

        row: dict = {"status": "Replied_With_Caveats", "request_type": "product_issue"}
        agent._sanitize_official_csv_columns(row)
        self.assertEqual(row["status"], "escalated")

    def test_synonym_request_type(self):
        import agent

        row: dict = {"status": "replied", "request_type": "feature"}
        agent._sanitize_official_csv_columns(row)
        self.assertEqual(row["request_type"], "feature_request")

    def test_garbage_request_type_invalid(self):
        import agent

        row: dict = {"status": "replied", "request_type": "nonsense_label"}
        agent._sanitize_official_csv_columns(row)
        self.assertEqual(row["request_type"], "invalid")


if __name__ == "__main__":
    unittest.main()
