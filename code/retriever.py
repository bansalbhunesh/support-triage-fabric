"""
BM25 retrieval over the hackathon Markdown corpus (+ distilled FAQ boosts).
Loads once per process; deterministic tokenization for stable rankings.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INDEX_VERSION = 5

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None  # type: ignore

from corpus import SUPPORT_CORPUS

try:
    from config import (
        CHUNK_BODY_MAX_CHARS,
        CHUNK_SPLIT_OVERLAP_STRIDE,
        DEFAULT_DOMAIN_HINT_BOOST,
        HYBRID_BM25_WEIGHT,
        HYBRID_OVERLAP_WEIGHT,
        HYBRID_SEMANTIC_WEIGHT,
        SEMANTIC_HASH_BUCKETS,
        SEMANTIC_HASH_DIM,
    )
except ImportError:  # pragma: no cover
    CHUNK_BODY_MAX_CHARS, CHUNK_SPLIT_OVERLAP_STRIDE = 2000, 260
    HYBRID_BM25_WEIGHT, HYBRID_OVERLAP_WEIGHT = 0.52, 0.30
    HYBRID_SEMANTIC_WEIGHT = 0.18
    SEMANTIC_HASH_DIM, SEMANTIC_HASH_BUCKETS = 256, 4096
    DEFAULT_DOMAIN_HINT_BOOST = 1.38

_PROJ_CACHE: dict[tuple[int, int], Any] = {}


def _bucket_index(token: str, n_buckets: int) -> int:
    digest = hashlib.blake2b(token.lower().encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_buckets


def _projection_matrix(dim: int, buckets: int) -> Any:
    if np is None:
        raise RuntimeError("numpy required for semantic retrieval")
    key = (dim, buckets)
    if key not in _PROJ_CACHE:
        rng = np.random.default_rng(4242 + dim * 17 + buckets * 3)
        _PROJ_CACHE[key] = (rng.standard_normal((dim, buckets)).astype(np.float32) * (1.0 / np.sqrt(buckets)))
    return _PROJ_CACHE[key]


def _embed_tokens(tokens: list[str], dim: int, buckets: int) -> Any:
    if np is None:
        return None
    P = _projection_matrix(dim, buckets)
    acc = np.zeros(dim, dtype=np.float32)
    for t in tokens:
        acc += P[:, _bucket_index(t, buckets)]
    n = float(np.linalg.norm(acc))
    return acc / n if n > 1e-8 else acc


_TOKEN_RE = re.compile(r"[a-z0-9/+]+", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]


@dataclass(frozen=True)
class Chunk:
    domain: str
    source_url: str
    title: str
    product_hint: str
    text: str

    def as_retrieval_blob(self) -> str:
        return f"{self.title}\n{self.product_hint}\n{self.text}"


def _parse_frontmatter(body: str) -> tuple[dict[str, str], str]:
    if not body.startswith("---"):
        return {}, body
    end = body.find("\n---", 3)
    if end == -1:
        return {}, body
    fm_raw = body[3:end].strip("\n")
    rest = body[end + 4 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in fm_raw.splitlines():
        m = re.match(r'^([\w_-]+):\s*"(.*)"\s*$', line.strip())
        if m:
            meta[m.group(1)] = m.group(2)
            continue
        m = re.match(r"^([\w_-]+):\s*(.+)\s*$", line.strip())
        if m and m.group(1) not in meta:
            meta[m.group(1)] = m.group(2).strip().strip("\"'")
    return meta, rest


def _split_chunks(md_body: str) -> list[str]:
    lines = md_body.splitlines()
    sections: list[str] = []
    buf: list[str] = []

    def flush():
        nonlocal buf
        if buf:
            s = "\n".join(buf).strip()
            if s and len(s) > 30:
                sections.append(s)
        buf = []

    for ln in lines:
        if ln.strip().startswith("#"):
            flush()
        buf.append(ln)
    flush()

    out: list[str] = []
    max_len = max(800, CHUNK_BODY_MAX_CHARS)
    overlap_stride = max(120, CHUNK_SPLIT_OVERLAP_STRIDE)
    step = max(200, max_len - overlap_stride)
    for s in sections:
        if len(s) <= max_len:
            out.append(s)
            continue
        for i in range(0, len(s), step):
            out.append(s[i : i + max_len])
    return out if out else [md_body.strip()[:max_len]]


def _infer_product_hint(rel_path: Path) -> str:
    parts = [p.replace("-", " ") for p in rel_path.parts[:-1] if p not in (".",)]
    return " / ".join(parts[-3:]) if parts else rel_path.parent.name


def _guess_url_from_path(domain: str, rel: Path) -> str:
    stem = rel.stem
    m = re.match(r"^(\d+)-(.+)$", stem)
    num, slug = (m.group(1), m.group(2)) if m else ("", stem)
    slug = slug.replace(" ", "-")
    if domain == "hackerrank" and num:
        return f"https://support.hackerrank.com/articles/{num}-{slug}"
    if domain == "claude" and num:
        return f"https://support.claude.com/en/articles/{num}-{slug}"
    if domain == "visa":
        return "https://www.visa.co.in/support.html"
    return ""


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    signatures: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        norm = re.sub(r"\s+", " ", (c.text or "").strip()[:520])
        dig = hashlib.sha256(norm.encode("utf-8", errors="replace")).hexdigest()[:24]
        sig = f"{c.domain}|{(c.source_url or '')}|{dig}"
        if sig in signatures:
            continue
        signatures.add(sig)
        out.append(c)
    return out


def load_chunks(data_root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for domain in ("hackerrank", "claude", "visa"):
        root = data_root / domain
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            meta, body = _parse_frontmatter(raw)
            source_url = meta.get("source_url") or meta.get("final_url") or ""
            title = meta.get("title") or path.stem.replace("-", " ")
            if not source_url:
                source_url = _guess_url_from_path(domain, path.relative_to(root))
            rel = path.relative_to(root)
            hint = _infer_product_hint(rel)
            for piece in _split_chunks(body):
                chunks.append(
                    Chunk(
                        domain=domain,
                        source_url=source_url or _guess_url_from_path(domain, rel),
                        title=title,
                        product_hint=hint,
                        text=piece,
                    )
                )

    for domain, blob in SUPPORT_CORPUS.items():
        for row in blob.get("faq", []):
            q, a = row.get("q", ""), row.get("a", "")
            src = row.get("source", "")
            if not q and not a:
                continue
            chunks.append(
                Chunk(
                    domain=domain,
                    source_url=src,
                    title=f"FAQ: {q[:120]}",
                    product_hint="help center summary",
                    text=f"Q: {q}\nA: {a}",
                )
            )
    return _dedupe_chunks(chunks)


def compute_corpus_fingerprint(data_root: Path) -> str:
    """Stable hash of corpus files + distilled FAQ overlay (invalidates cache on change)."""
    h = hashlib.sha256()
    h.update(
        f"index_v={INDEX_VERSION}|algo=bm25_overlap_sem|sw={HYBRID_SEMANTIC_WEIGHT}|"
        f"dim={SEMANTIC_HASH_DIM}|bk={SEMANTIC_HASH_BUCKETS}\n".encode()
    )
    for domain in ("hackerrank", "claude", "visa"):
        root = data_root / domain
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            try:
                rel = path.relative_to(root).as_posix()
                st = path.stat()
                h.update(f"{domain}/{rel}\n{st.st_size}\n{st.st_mtime_ns}\n".encode())
            except OSError:
                continue
    for dom in sorted(SUPPORT_CORPUS.keys()):
        blob = SUPPORT_CORPUS[dom]
        h.update(dom.encode())
        for row in blob.get("faq", []):
            h.update(
                f"{row.get('q', '')}\n{row.get('a', '')}\n{row.get('source', '')}\n".encode(
                    "utf-8", errors="replace"
                )
            )
    return h.hexdigest()


def _index_cache_path(cache_dir: Path, fingerprint: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    short = fingerprint[:24]
    return cache_dir / f"bm25_index_v{INDEX_VERSION}_{short}.pkl.gz"


def _try_load_index_cache(path: Path, fingerprint: str) -> tuple[list[Chunk], list[list[str]]] | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rb") as gz:
            payload: dict[str, Any] = pickle.load(gz)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError):
        return None
    if payload.get("version") != INDEX_VERSION or payload.get("fingerprint") != fingerprint:
        return None
    chunks = payload.get("chunks")
    tokenized = payload.get("tokenized")
    if not isinstance(chunks, list) or not isinstance(tokenized, list):
        return None
    return chunks, tokenized


def _save_index_cache(path: Path, fingerprint: str, chunks: list[Chunk], tokenized: list[list[str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with gzip.open(tmp, "wb", compresslevel=6) as gz:
            pickle.dump(
                {
                    "version": INDEX_VERSION,
                    "fingerprint": fingerprint,
                    "chunks": chunks,
                    "tokenized": tokenized,
                },
                gz,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass


class CorpusRetriever:
    def __init__(self, data_root: Path, cache_dir: Path | None = None):
        self.data_root = data_root
        self.cache_dir = cache_dir or (data_root.parent / "logs")
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []
        self._semantic_mat: Any = None
        self._semantic_materialized = False
        self.last_index_event: str = "uninitialized"

    def _materialize_semantic_matrix(self) -> None:
        if self._semantic_materialized:
            return
        self._semantic_materialized = True
        if (
            np is None
            or HYBRID_SEMANTIC_WEIGHT <= 1e-9
            or not self._tokenized
        ):
            self._semantic_mat = None
            return
        dim, buckets = SEMANTIC_HASH_DIM, SEMANTIC_HASH_BUCKETS
        rows: list[Any] = []
        for doc in self._tokenized:
            rows.append(_embed_tokens(doc, dim, buckets))
        self._semantic_mat = np.stack(rows, axis=0)

    def ensure_built(self) -> None:
        if self._bm25 is not None or (self._tokenized and not BM25Okapi):
            return

        fingerprint = compute_corpus_fingerprint(self.data_root)
        rebuild = os.environ.get("SUPPORT_AGENT_REBUILD_INDEX", "").lower() in ("1", "true", "yes", "force")
        cache_path = _index_cache_path(self.cache_dir, fingerprint)

        if not rebuild:
            cached = _try_load_index_cache(cache_path, fingerprint)
            if cached is not None:
                self.chunks, self._tokenized = cached
                self.last_index_event = f"cache_hit:{cache_path.name}"
                if BM25Okapi is None or not self._tokenized:
                    self._bm25 = None
                else:
                    self._bm25 = BM25Okapi(self._tokenized)
                return

        self.chunks = load_chunks(self.data_root)
        self._tokenized = [tokenize(c.as_retrieval_blob()) for c in self.chunks]
        self.last_index_event = "built_fresh"
        if BM25Okapi is None or not self._tokenized:
            self._bm25 = None
        else:
            self._bm25 = BM25Okapi(self._tokenized)

        if not rebuild:
            try:
                _save_index_cache(cache_path, fingerprint, self.chunks, self._tokenized)
                self.last_index_event = f"built_fresh_saved:{cache_path.name}"
            except OSError:
                self.last_index_event = "built_fresh_save_failed"

    def search(
        self,
        query: str,
        domain_hint: str | None,
        top_k: int = 8,
        domain_boost: float | None = None,
    ) -> tuple[list[tuple[Chunk, float]], dict[str, float]]:
        self.ensure_built()
        self._materialize_semantic_matrix()
        qtok = tokenize(query)
        stats: dict[str, float] = {"query_tokens": float(len(qtok))}

        if not qtok or not self.chunks:
            return [], stats

        boost = DEFAULT_DOMAIN_HINT_BOOST if domain_boost is None else domain_boost

        if self._bm25 is not None:
            scores = self._bm25.get_scores(qtok)
        else:
            scores = []
            for doc in self._tokenized:
                qset = set(qtok)
                scores.append(sum(1 for t in doc if t in qset))

        raw_lex = list(scores)
        if domain_hint and domain_hint in {"hackerrank", "claude", "visa"}:
            for i, ch in enumerate(self.chunks):
                if ch.domain == domain_hint:
                    raw_lex[i] *= boost

        qset = set(qtok)
        overlap_scores = [sum(1 for t in doc if t in qset) for doc in self._tokenized]

        window = min(len(raw_lex), max(380, top_k * 52))
        cand_idx = sorted(range(len(raw_lex)), key=lambda i: raw_lex[i], reverse=True)[:window]

        mx_b = max((raw_lex[i] for i in cand_idx), default=1.0) or 1.0
        mx_o = max((overlap_scores[i] for i in cand_idx), default=1.0) or 1.0

        sem_all: Any = None
        sem_window: list[float] | None = None
        mx_s = 1.0
        if self._semantic_mat is not None and HYBRID_SEMANTIC_WEIGHT > 1e-9 and np is not None:
            q_emb = _embed_tokens(qtok, SEMANTIC_HASH_DIM, SEMANTIC_HASH_BUCKETS)
            sem_all = self._semantic_mat @ q_emb
            sem_window = [max(0.0, float(sem_all[i])) for i in cand_idx]
            mx_s = max(sem_window, default=1.0) or 1.0

        fused: dict[int, float] = {}
        for j, i in enumerate(cand_idx):
            b = raw_lex[i] / mx_b
            o = overlap_scores[i] / mx_o
            piece = HYBRID_BM25_WEIGHT * b + HYBRID_OVERLAP_WEIGHT * o
            if sem_window is not None:
                sem_n = sem_window[j] / mx_s
                piece += HYBRID_SEMANTIC_WEIGHT * sem_n
            fused[i] = piece

        reranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        out = [(self.chunks[i], s) for i, s in reranked]

        top1 = reranked[0][1] if reranked else 0.0
        top2 = reranked[1][1] if len(reranked) > 1 else 0.0
        stats.update(
            {
                "top1": float(top1),
                "margin": float((top1 - top2) / (top1 + 1e-6)) if top1 else 0.0,
                "mean_top3": float(sum(s for _, s in reranked[:3]) / min(3, len(reranked)))
                if reranked
                else 0.0,
            }
        )
        if sem_all is not None and reranked:
            i0 = reranked[0][0]
            s0 = max(0.0, float(sem_all[i0]))
            s1 = max(0.0, float(sem_all[reranked[1][0]])) if len(reranked) > 1 else 0.0
            stats["semantic_top1"] = float(s0)
            stats["semantic_margin"] = float((s0 - s1) / (s0 + 1e-6)) if s0 else 0.0
            stats["semantic_enabled"] = 1.0
        else:
            stats["semantic_top1"] = 0.0
            stats["semantic_margin"] = 0.0
            stats["semantic_enabled"] = 0.0
        return out, stats


def retrieval_confidence_ok(stats: dict[str, float], has_api: bool) -> tuple[bool, str]:
    """Telemetry-style signal (non-blocking unless paired with synthesis risk)."""
    top1 = stats.get("top1", 0.0)
    margin = stats.get("margin", 0.0)
    qt = stats.get("query_tokens", 0.0)

    if qt < 3:
        return False, "very_short_query"

    if BM25Okapi is None:
        if top1 < 2.0:
            return False, "keyword_overlap_low_fallback"
        return True, "keyword_fallback_ok"

    if top1 < 6.8 and margin < 0.035:
        return False, "weak_top_match_low_evidence"

    return True, "ok"


def should_force_escalate_from_retrieval(stats: dict[str, float], has_api: bool) -> tuple[bool, str]:
    """Conservative grounding gate — used when relying on verbatim snippet extraction."""
    top1 = stats.get("top1", 0.0)
    margin = stats.get("margin", 0.0)
    qt = stats.get("query_tokens", 0.0)

    if qt < 3:
        return True, "very_short_query"

    if BM25Okapi is None:
        if top1 < 1.6:
            return True, "keyword_overlap_critically_low"
        return False, "keyword_fallback_ok"

    # High absolute relevance with microscopic margin usually means multiple near-duplicates
    if top1 >= 58.0 and margin < 0.015:
        return False, "near_dup_top_docs"

    if top1 < 4.9 or margin < 0.015:
        return True, "retrieval_uncertainty"

    if not has_api and (top1 < 7.2 or margin < 0.021):
        return True, "low_margin_without_llm"

    return False, ""
