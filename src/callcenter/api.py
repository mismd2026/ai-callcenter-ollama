"""HTTP API (FastAPI).

    POST /v1/tenants/{tenant}/calls                      start a call session
    POST /v1/tenants/{tenant}/calls/{call_id}/text       one text turn  (JSON)
    POST /v1/tenants/{tenant}/calls/{call_id}/audio      one voice turn (WAV in / WAV out)
    GET  /v1/tenants/{tenant}/calls/{call_id}/handoff    operator brief after escalation
    GET  /healthz                                        liveness
    GET  /metrics                                        Prometheus text format

The telephony side (Asterisk / Twilio-compatible SIP gateway) streams caller
audio per utterance to the /audio endpoint.
"""

from __future__ import annotations

import base64
import uuid
from collections import Counter
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from .adapters import SpeechToText, TextToSpeech
from .escalation import ConversationState, Layer
from .handoff import HandoffSummarizer, HandoffSummary
from .pipeline import TurnResult, VoicePipeline
from .tenants import Tenant, TenantRegistry


class TextTurn(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    include_audio: bool = False


class TurnOut(BaseModel):
    transcript: str
    reply: str
    layer: int
    reason: str
    sources: list[str]
    latency_ms: dict[str, float]
    audio_wav_base64: str | None = None
    handoff_url: str | None = None


class HandoffOut(BaseModel):
    status: str  # "pending" | "ready"
    summary: str | None = None
    reason: str | None = None
    last_utterance: str | None = None
    transcript: list[dict[str, str]] = []
    faq_sources: list[str] = []
    generated_by: str | None = None


def create_app(
    registry: TenantRegistry,
    stt: SpeechToText,
    tts: TextToSpeech,
    summarizer: HandoffSummarizer | None = None,
) -> FastAPI:
    app = FastAPI(title="AI Call Center", version="0.1.0")
    metrics: Counter[str] = Counter()
    # call_id -> brief; None means "escalated, summary being generated".
    handoffs: dict[str, HandoffSummary | None] = {}

    def _tenant(tenant_id: str) -> Tenant:
        tenant = registry.get(tenant_id)
        if tenant is None:
            raise HTTPException(404, "unknown tenant")
        return tenant

    def _session(tenant: Tenant, call_id: str) -> ConversationState:
        state = tenant.sessions.get(call_id)
        if state is None:
            raise HTTPException(404, "unknown call")
        return state

    def _record(tenant: Tenant, result: TurnResult) -> None:
        layer = int(result.decision.layer)
        metrics[f'callcenter_turns_total{{tenant="{tenant.id}",layer="{layer}"}}'] += 1

    def _schedule_handoff(
        tasks: BackgroundTasks,
        tenant: Tenant,
        call_id: str,
        state: ConversationState,
        result: TurnResult,
    ) -> str | None:
        """On escalation, build the operator brief after the reply has been sent,
        so the caller hears "おつなぎします" without waiting for the summary LLM."""
        if result.decision.layer is not Layer.HUMAN:
            return None
        key = f"{tenant.id}/{call_id}"
        handoffs[key] = None
        history = list(state.history)
        sources = list(state.cited_sources)
        reason = result.decision.reason
        llm_summarizer = summarizer or HandoffSummarizer(tenant.policy.llm)

        def build() -> None:
            handoffs[key] = llm_summarizer.summarize(history, reason, sources)

        tasks.add_task(build)
        return f"/v1/tenants/{tenant.id}/calls/{call_id}/handoff"

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus() -> str:
        lines = ["# TYPE callcenter_turns_total counter"]
        lines += [f"{k} {v}" for k, v in sorted(metrics.items())]
        return "\n".join(lines) + "\n"

    @app.post("/v1/tenants/{tenant_id}/calls", status_code=201)
    def start_call(tenant_id: str) -> dict[str, str]:
        tenant = _tenant(tenant_id)
        call_id = uuid.uuid4().hex
        tenant.sessions[call_id] = ConversationState()
        return {"call_id": call_id}

    @app.post("/v1/tenants/{tenant_id}/calls/{call_id}/text", response_model=TurnOut)
    def text_turn(tenant_id: str, call_id: str, body: TextTurn, tasks: BackgroundTasks) -> TurnOut:
        tenant = _tenant(tenant_id)
        state = _session(tenant, call_id)
        result = VoicePipeline(stt, tenant.policy, tts).handle_text(body.text, state)
        _record(tenant, result)
        handoff_url = _schedule_handoff(tasks, tenant, call_id, state, result)
        d = result.decision
        audio_b64 = base64.b64encode(result.audio).decode() if body.include_audio else None
        return TurnOut(
            transcript=result.transcript,
            reply=d.reply,
            layer=int(d.layer),
            reason=d.reason,
            sources=d.sources,
            latency_ms=result.latency_ms,
            audio_wav_base64=audio_b64,
            handoff_url=handoff_url,
        )

    @app.post("/v1/tenants/{tenant_id}/calls/{call_id}/audio")
    async def audio_turn(
        tenant_id: str, call_id: str, file: UploadFile, tasks: BackgroundTasks
    ) -> Response:
        tenant = _tenant(tenant_id)
        state = _session(tenant, call_id)
        audio = await file.read()
        if not audio:
            raise HTTPException(400, "empty audio")
        result = VoicePipeline(stt, tenant.policy, tts).handle_audio(audio, state)
        _record(tenant, result)
        headers = {
            "X-Escalation-Layer": str(int(result.decision.layer)),
            "X-Transcript": quote(result.transcript),
            "X-Latency-Ms": f"{result.latency_ms.get('total', 0):.0f}",
        }
        handoff_url = _schedule_handoff(tasks, tenant, call_id, state, result)
        if handoff_url:
            headers["X-Handoff-Url"] = handoff_url
        return Response(content=result.audio, media_type="audio/wav", headers=headers)

    @app.get("/v1/tenants/{tenant_id}/calls/{call_id}/handoff", response_model=HandoffOut)
    def get_handoff(tenant_id: str, call_id: str) -> HandoffOut:
        tenant = _tenant(tenant_id)
        _session(tenant, call_id)
        key = f"{tenant.id}/{call_id}"
        if key not in handoffs:
            raise HTTPException(404, "call has not been handed off")
        brief = handoffs[key]
        if brief is None:
            return HandoffOut(status="pending")
        return HandoffOut(
            status="ready",
            summary=brief.summary,
            reason=brief.reason,
            last_utterance=brief.last_utterance,
            transcript=brief.transcript,
            faq_sources=brief.faq_sources,
            generated_by=brief.generated_by,
        )

    return app
