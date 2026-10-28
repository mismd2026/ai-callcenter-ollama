"""Adapters for the three self-hosted model servers: STT, LLM and TTS.

Each adapter implements a small Protocol so that the call pipeline can be tested
with in-memory fakes and so that an engine can be swapped (e.g. Whisper -> ReazonSpeech)
without touching business logic.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Protocol

import httpx

# --------------------------------------------------------------------------- protocols


class SpeechToText(Protocol):
    def transcribe(self, audio: bytes, language: str = "ja") -> str: ...


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> bytes: ...


# --------------------------------------------------------------------------- STT


class WhisperSTT:
    """faster-whisper (CTranslate2) running on the same node as the API pod."""

    def __init__(self, model_size: str = "medium", device: str = "auto") -> None:
        from faster_whisper import WhisperModel  # optional dependency

        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: bytes, language: str = "ja") -> str:
        segments, _info = self._model.transcribe(
            io.BytesIO(audio),
            language=language,
            vad_filter=True,  # drop silence -> lower latency and fewer hallucinations
            beam_size=1,
        )
        return "".join(seg.text for seg in segments).strip()


# --------------------------------------------------------------------------- LLM


@dataclass
class OllamaChat:
    """Thin client for Ollama's /api/chat endpoint (non-streaming)."""

    base_url: str
    model: str
    timeout: float = 60.0

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def chat_json(self, messages: list[dict[str, str]]) -> dict:
        """Ask the model for strict JSON output (used by the escalation judge)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return json.loads(resp.json()["message"]["content"])


# --------------------------------------------------------------------------- TTS


@dataclass
class VoicevoxTTS:
    """VOICEVOX engine: /audio_query -> /synthesis, returns WAV bytes."""

    base_url: str
    speaker: int = 3
    speed_scale: float = 1.1
    timeout: float = 30.0

    def synthesize(self, text: str) -> bytes:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            q = client.post("/audio_query", params={"text": text, "speaker": self.speaker})
            q.raise_for_status()
            query = q.json()
            query["speedScale"] = self.speed_scale
            s = client.post("/synthesis", params={"speaker": self.speaker}, json=query)
            s.raise_for_status()
            return s.content
