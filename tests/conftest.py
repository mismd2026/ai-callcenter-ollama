"""Test doubles: deterministic stand-ins for the model servers so the whole
call flow runs in CI in milliseconds, without GPU or network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from callcenter.escalation import ConversationState, EscalationPolicy
from callcenter.rag import FaqIndex, load_faq

KEYWORDS = ["診療時間", "駐車場", "予約", "カード", "診断書"]
FAQ_DIR = Path(__file__).resolve().parents[1] / "data" / "faq" / "demo-clinic"


class KeywordEmbedder:
    """Bag-of-keywords embedder. Cosine similarity is fully predictable:
    one shared keyword -> 1.0, query spanning two topics -> ~0.71, none -> 0.0."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.extend(texts)
        out = np.zeros((len(texts), len(KEYWORDS) + 1), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, kw in enumerate(KEYWORDS):
                out[i, j] = t.count(kw)
            if not out[i].any():
                out[i, -1] = 1.0 if t.startswith("query: ") else 0.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class FakeLLM:
    def __init__(self, reply: str = "駐車場は5台分あり、予約は前日17時まで変更できます。") -> None:
        self.reply = reply
        self.messages: list[list[dict[str, str]]] = []

    def chat(self, messages, *, temperature: float = 0.2) -> str:
        self.messages.append(messages)
        return self.reply


class FakeTTS:
    def synthesize(self, text: str) -> bytes:
        return b"RIFF" + text.encode()


class FakeSTT:
    """Treats the uploaded 'audio' as UTF-8 text."""

    def transcribe(self, audio: bytes, language: str = "ja") -> str:
        return audio.decode()


@pytest.fixture
def embedder() -> KeywordEmbedder:
    return KeywordEmbedder()


@pytest.fixture
def faq():
    return load_faq(FAQ_DIR)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def policy(embedder, faq, llm) -> EscalationPolicy:
    index = FaqIndex(embedder)
    index.add(faq)
    return EscalationPolicy(index, llm)


@pytest.fixture
def state() -> ConversationState:
    return ConversationState()
