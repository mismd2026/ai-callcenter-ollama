"""Retrieval-augmented FAQ search.

multilingual-e5 models expect the prefixes ``query: `` and ``passage: ``.
Forgetting them measurably lowers recall, so the prefixes are applied here
rather than by callers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) array of L2-normalised vectors."""
        ...


class E5Embedder:
    """sentence-transformers wrapper for intfloat/multilingual-e5-*."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base") -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


@dataclass(frozen=True)
class FaqEntry:
    id: str
    question: str
    answer: str
    category: str = "general"


@dataclass(frozen=True)
class Hit:
    entry: FaqEntry
    score: float


def load_faq(path: str | Path) -> list[FaqEntry]:
    """Load FAQ entries from a JSONL file or a directory of JSONL files."""
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    entries: list[FaqEntry] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(FaqEntry(**json.loads(line)))
    return entries


class FaqIndex:
    """In-memory cosine index. For production volumes the same interface is
    backed by pgvector (see db/schema.sql)."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._entries: list[FaqEntry] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entries: Iterable[FaqEntry]) -> None:
        new = list(entries)
        if not new:
            return
        # Index question and answer together so paraphrased questions still match.
        vecs = self._embedder.embed([f"passage: {e.question}\n{e.answer}" for e in new])
        self._entries.extend(new)
        self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])

    def search(self, query: str, k: int = 3) -> list[Hit]:
        if self._matrix is None:
            return []
        q = self._embedder.embed([f"query: {query}"])[0]
        scores = self._matrix @ q
        top = np.argsort(-scores)[:k]
        return [Hit(self._entries[i], float(scores[i])) for i in top]
