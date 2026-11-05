"""Adapters for the three self-hosted model servers: STT, LLM and TTS.

Each adapter implements a small Protocol so that the call pipeline can be tested
with in-memory fakes and so that an engine can be swapped (e.g. Whisper -> ReazonSpeech)
without touching business logic.
"""

from __future__ import annotations

import io
import json
import wave
from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np

# --------------------------------------------------------------------------- protocols


class SpeechToText(Protocol):
    def transcribe(self, audio: bytes, language: str = "ja") -> str: ...


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> bytes: ...


# --------------------------------------------------------------------------- STT

WHISPER_SAMPLE_RATE = 16000


class TranscriptionError(RuntimeError):
    """Speech recognition failed (undecodable audio or an engine error)."""


def decode_wav(audio: bytes, target_rate: int = WHISPER_SAMPLE_RATE) -> np.ndarray | None:
    """Decode PCM WAV into mono float32 at `target_rate`, without ffmpeg/PyAV.

    Telephony gateways and VOICEVOX both send PCM WAV, so decoding it here keeps
    speech recognition independent of the PyAV version faster-whisper happens to
    be installed with. Returns None for anything that is not PCM WAV.
    """
    try:
        with wave.open(io.BytesIO(audio)) as w:
            channels, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError):
        return None
    if width == 1:  # 8-bit PCM is unsigned
        pcm = (np.frombuffer(frames, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        pcm = np.frombuffer(frames, "<i2").astype(np.float32) / 32768
    elif width == 4:
        pcm = np.frombuffer(frames, "<i4").astype(np.float32) / 2147483648
    else:
        return None
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if rate != target_rate and len(pcm):
        n = max(1, round(len(pcm) * target_rate / rate))
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm)
    return pcm.astype(np.float32)


class WhisperSTT:
    """faster-whisper (CTranslate2) running on the same node as the API pod."""

    def __init__(self, model_size: str = "medium", device: str = "auto") -> None:
        from faster_whisper import WhisperModel  # optional dependency

        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: bytes, language: str = "ja") -> str:
        pcm = decode_wav(audio)
        # PCM WAV is decoded here; other formats fall back to faster-whisper's own decoder.
        source = pcm if pcm is not None else io.BytesIO(audio)
        try:
            segments, _info = self._model.transcribe(
                source,
                language=language,
                vad_filter=True,  # drop silence -> lower latency and fewer hallucinations
                beam_size=1,
            )
            return "".join(seg.text for seg in segments).strip()
        except Exception as e:  # surface as a handled error, not a 500
            raise TranscriptionError(f"{type(e).__name__}: {e}") from e


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
