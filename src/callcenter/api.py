"""HTTP API (FastAPI).

    POST /v1/tenants/{tenant}/calls                      start a call session
    POST /v1/tenants/{tenant}/calls/{call_id}/text       one text turn  (JSON)
    POST /v1/tenants/{tenant}/calls/{call_id}/audio      one voice turn (WAV in / WAV out)
    DELETE /v1/tenants/{tenant}/calls/{call_id}          end the call, discard its conversation
    GET  /v1/tenants/{tenant}/calls/{call_id}/handoff    operator brief after escalation
    DELETE /v1/tenants/{tenant}/calls/{call_id}/handoff  operator acknowledged, discard brief

Conversation data is personal data and lives in memory only. It is discarded
when the call ends, when a call is idle longer than `call_idle_timeout_s`, and
operator briefs after they are acknowledged or `handoff_ttl_s` has passed.
    GET  /healthz                                        liveness
    GET  /metrics                                        Prometheus text format

The telephony side (Asterisk / Twilio-compatible SIP gateway) streams caller
audio per utterance to the /audio endpoint.
"""

from __future__ import annotations

import base64
import time
import uuid
from collections import Counter
from collections.abc import Callable
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


class CallEndOut(BaseModel):
    call_id: str
    turns: int
    final_layer: int | None
    handoff_pending: bool


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
    *,
    call_idle_timeout_s: float = 600.0,
    handoff_ttl_s: float = 1800.0,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    app = FastAPI(title="AI Call Center", version="0.1.0")
    metrics: Counter[str] = Counter()
    # "tenant/call_id" -> brief; None means "escalated, summary being generated".
    handoffs: dict[str, HandoffSummary | None] = {}
    handoff_created: dict[str, float] = {}
    # "tenant/call_id" -> last activity time, for idle-call expiry.
    last_seen: dict[str, float] = {}
    last_layer: dict[str, int] = {}

    def _key(tenant_id: str, call_id: str) -> str:
        return f"{tenant_id}/{call_id}"

    def _drop_call(key: str, why: str) -> ConversationState | None:
        tenant_id, call_id = key.split("/", 1)
        last_seen.pop(key, None)
        last_layer.pop(key, None)
        tenant = registry.get(tenant_id)
        state = tenant.sessions.pop(call_id, None) if tenant else None
        if state is not None:
            metrics[f'callcenter_calls_closed_total{{tenant="{tenant_id}",why="{why}"}}'] += 1
        return state

    def _drop_handoff(key: str) -> None:
        handoffs.pop(key, None)
        handoff_created.pop(key, None)

    @app.middleware("http")
    async def sweep_expired(request, call_next):
        """Lazy expiry on every request: no background thread to manage, and
        nothing outlives its deadline by more than one request interval."""
        now = clock()
        for key in [k for k, t in last_seen.items() if now - t > call_idle_timeout_s]:
            _drop_call(key, "idle")
        for key in [k for k, t in handoff_created.items() if now - t > handoff_ttl_s]:
            _drop_handoff(key)
        return await call_next(request)

    def _tenant(tenant_id: str) -> Tenant:
        tenant = registry.get(tenant_id)
        if tenant is None:
            raise HTTPException(404, "unknown tenant")
        return tenant

    def _session(tenant: Tenant, call_id: str) -> ConversationState:
        state = tenant.sessions.get(call_id)
        if state is None:
            raise HTTPException(404, "unknown call")
        last_seen[_key(tenant.id, call_id)] = clock()
        return state

    def _record(tenant: Tenant, call_id: str, result: TurnResult) -> None:
        layer = int(result.decision.layer)
        last_layer[_key(tenant.id, call_id)] = layer
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
        key = _key(tenant.id, call_id)
        handoffs[key] = None
        handoff_created[key] = clock()
        history = list(state.history)
        sources = list(state.cited_sources)
        reason = result.decision.reason
        llm_summarizer = summarizer or HandoffSummarizer(tenant.policy.llm)

        def build() -> None:
            brief = llm_summarizer.summarize(history, reason, sources)
            if key in handoffs:  # not acknowledged / expired in the meantime
                handoffs[key] = brief

        tasks.add_task(build)
        return f"/v1/tenants/{tenant.id}/calls/{call_id}/handoff"

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus() -> str:
        lines = ["# TYPE callcenter_turns_total counter"]
        lines += [
            f"{k} {v}" for k, v in sorted(metrics.items()) if k.startswith("callcenter_turns")
        ]
        lines += ["# TYPE callcenter_calls_closed_total counter"]
        lines += [
            f"{k} {v}" for k, v in sorted(metrics.items()) if k.startswith("callcenter_calls")
        ]
        lines += [
            "# TYPE callcenter_active_calls gauge",
            f"callcenter_active_calls {len(last_seen)}",
        ]
        lines += [
            "# TYPE callcenter_pending_handoffs gauge",
            f"callcenter_pending_handoffs {len(handoffs)}",
        ]
        return "\n".join(lines) + "\n"

    @app.post("/v1/tenants/{tenant_id}/calls", status_code=201)
    def start_call(tenant_id: str) -> dict[str, str]:
        tenant = _tenant(tenant_id)
        call_id = uuid.uuid4().hex
        tenant.sessions[call_id] = ConversationState()
        last_seen[_key(tenant.id, call_id)] = clock()
        return {"call_id": call_id}

    @app.delete("/v1/tenants/{tenant_id}/calls/{call_id}", response_model=CallEndOut)
    def end_call(tenant_id: str, call_id: str) -> CallEndOut:
        """Called by the telephony gateway on hang-up or transfer. The conversation
        is discarded; an operator brief, if any, stays until acknowledged or TTL."""
        tenant = _tenant(tenant_id)
        key = _key(tenant.id, call_id)
        layer = last_layer.get(key)
        state = _drop_call(key, "ended")
        if state is None:
            raise HTTPException(404, "unknown call")
        return CallEndOut(
            call_id=call_id,
            turns=len(state.recent_queries),
            final_layer=layer,
            handoff_pending=key in handoffs,
        )

    @app.post("/v1/tenants/{tenant_id}/calls/{call_id}/text", response_model=TurnOut)
    def text_turn(tenant_id: str, call_id: str, body: TextTurn, tasks: BackgroundTasks) -> TurnOut:
        tenant = _tenant(tenant_id)
        state = _session(tenant, call_id)
        result = VoicePipeline(stt, tenant.policy, tts).handle_text(body.text, state)
        _record(tenant, call_id, result)
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
        _record(tenant, call_id, result)
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
        # The brief must stay readable after the AI leg of the call has ended.
        tenant = _tenant(tenant_id)
        key = _key(tenant.id, call_id)
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

    @app.delete("/v1/tenants/{tenant_id}/calls/{call_id}/handoff", status_code=204)
    def ack_handoff(tenant_id: str, call_id: str) -> Response:
        tenant = _tenant(tenant_id)
        key = _key(tenant.id, call_id)
        if key not in handoffs:
            raise HTTPException(404, "call has not been handed off")
        _drop_handoff(key)
        return Response(status_code=204)

    return app
