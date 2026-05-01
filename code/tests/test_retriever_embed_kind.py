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


if __name__ == "__main__":
    unittest.main()
