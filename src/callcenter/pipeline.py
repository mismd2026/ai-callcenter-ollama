"""Voice turn pipeline: caller audio -> STT -> policy -> TTS -> reply audio.

Per-stage latency is recorded for every turn because, on a phone line, total
response time (not model quality) is what callers notice first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .adapters import SpeechToText, TextToSpeech
from .escalation import ConversationState, Decision, EscalationPolicy


@dataclass
class TurnResult:
    transcript: str
    decision: Decision
    audio: bytes
    latency_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class VoicePipeline:
    stt: SpeechToText
    policy: EscalationPolicy
    tts: TextToSpeech

    def handle_text(self, text: str, state: ConversationState) -> TurnResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        decision = self.policy.decide(text, state)
        timings["policy"] = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        audio = self.tts.synthesize(decision.reply)
        timings["tts"] = (time.perf_counter() - t1) * 1000
        return TurnResult(text, decision, audio, timings)

    def handle_audio(self, audio: bytes, state: ConversationState) -> TurnResult:
        t0 = time.perf_counter()
        transcript = self.stt.transcribe(audio)
        stt_ms = (time.perf_counter() - t0) * 1000
        result = self.handle_text(transcript, state)
        result.latency_ms = {"stt": stt_ms, **result.latency_ms}
        result.latency_ms["total"] = sum(result.latency_ms.values())
        return result
