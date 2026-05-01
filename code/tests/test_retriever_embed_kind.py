"""Dense embedding backend routing (no network)."""

from __future__ import annotations

import io
import os
import pathlib
import sys
import unittest
import urllib.error
import urllib.request
from unittest import mock

_CODE = pathlib.Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))


class TestEmbedKind(unittest.TestCase):
    def tearDown(self) -> None:
        for k in (
            "SUPPORT_AGENT_EMBEDDING_BACKEND",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ):
            os.environ.pop(k, None)

    def test_defaults_to_none(self):
        import importlib

        import config as cfg
        import retriever as ret

        importlib.reload(cfg)
        importlib.reload(ret)
        self.assertEqual(ret._dense_embed_kind(), "none")

    def test_openai_without_key_is_none(self):
        import importlib

        import config as cfg
        import retriever as ret

        os.environ["SUPPORT_AGENT_EMBEDDING_BACKEND"] = "openai"
        os.environ.pop("OPENAI_API_KEY", None)
        importlib.reload(cfg)
        importlib.reload(ret)
        self.assertEqual(ret._dense_embed_kind(), "none")

    def test_openai_with_key(self):
        import importlib

        import config as cfg
        import retriever as ret

        os.environ["SUPPORT_AGENT_EMBEDDING_BACKEND"] = "openai"
        os.environ["OPENAI_API_KEY"] = "sk-test"
        importlib.reload(cfg)
        importlib.reload(ret)
        self.assertEqual(ret._dense_embed_kind(), "openai")

    def test_openai_embed_http_error_surfaces_body(self):
        import importlib

        import config as cfg
        import retriever as ret

        os.environ["SUPPORT_AGENT_EMBEDDING_BACKEND"] = "openai"
        os.environ["OPENAI_API_KEY"] = "sk-test"
        importlib.reload(cfg)
        importlib.reload(ret)

        fp = io.BytesIO(b'{"error":{"message":"Incorrect API key"}}')
        err = urllib.error.HTTPError(
            "https://api.openai.com/v1/embeddings", 401, "Unauthorized", {}, fp
        )
        with mock.patch.object(urllib.request, "urlopen", side_effect=err):
            with self.assertRaises(RuntimeError) as ctx:
                ret._openai_embed_texts(["hello"])
        self.assertIn("401", str(ctx.exception))
        self.assertIn("Incorrect API key", str(ctx.exception))

    def test_dense_build_failure_detail(self):
        import retriever as ret

        self.assertEqual(ret._dense_build_failure_detail(ValueError("bad")), "bad")
        self.assertNotIn("|", ret._dense_build_failure_detail(RuntimeError("a|b")))
        long = "z" * 400
        d = ret._dense_build_failure_detail(RuntimeError(long))
        self.assertLessEqual(len(d), 220)
        self.assertTrue(d.endswith("…"))


class TestDenseQueryFailure(unittest.TestCase):
    def test_search_dense_query_failure_falls_back_lexical(self):
        import numpy as np

        import retriever as ret

        repo = pathlib.Path(__file__).resolve().parents[2]
        data = repo / "data"
        if not data.is_dir():
            self.skipTest("no data directory")
        r = ret.CorpusRetriever(data)
        r.ensure_built()
        if not r.chunks or not r._tokenized:
            self.skipTest("empty index")
        r._dense_mat = np.zeros((len(r.chunks), 3), dtype=np.float32)
        with mock.patch.object(ret, "_embed_dense_query_vector", side_effect=RuntimeError("network_down")):
            ranked, stats = r.search(
                "how do I reset my password for hacker rank account login",
                "hackerrank",
                top_k=4,
            )
        self.assertIn("network_down", stats.get("dense_query_error", ""))
        self.assertEqual(float(stats.get("dense_enabled", 1)), 0.0)
        self.assertTrue(ranked)


if __name__ == "__main__":
    unittest.main()
