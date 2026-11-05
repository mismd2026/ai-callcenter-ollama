"""Production entrypoint: wires real model servers into the API.

uvicorn callcenter.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .adapters import OllamaChat, VoicevoxTTS, WhisperSTT
from .api import create_app
from .config import Settings, assert_data_sovereignty
from .deployment import RecordingStore, build_manifest, save_manifest
from .rag import E5Embedder, load_faq
from .records import Outbox, RecordingConfig
from .tenants import TenantRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("callcenter")


def build_app():
    s = Settings()
    assert_data_sovereignty(s)

    state_dir = Path(s.state_dir)
    manifest = build_manifest(
        public_url=s.public_url or None,
        ca_file=s.public_ca_file or None,
        state_dir=state_dir,
        faq_dir=s.faq_dir,
        bind=s.bind,
    )
    log.info("deployment recorded at %s", save_manifest(manifest, state_dir))
    store = RecordingStore(state_dir)

    registry = TenantRegistry(
        embedder=E5Embedder(s.embedding_model),
        llm=OllamaChat(s.ollama_url, s.ollama_model),
        direct_threshold=s.faq_direct_threshold,
        context_threshold=s.rag_context_threshold,
        max_ai_turns=s.max_ai_turns_without_resolution,
        max_history_turns=s.max_history_turns,
    )
    # data/faq/<tenant-id>/tenant.json + *.jsonl
    for tenant_dir in sorted(p for p in Path(s.faq_dir).iterdir() if p.is_dir()):
        meta = json.loads((tenant_dir / "tenant.json").read_text(encoding="utf-8"))
        # A token issued at runtime (state dir) overrides tenant.json.
        recording = store.load(tenant_dir.name) or RecordingConfig.from_dict(meta.get("recording"))
        tenant = registry.register(tenant_dir.name, meta["name"], load_faq(tenant_dir), recording)
        log.info(
            "tenant %s loaded with %d FAQ entries (recording %s)",
            tenant.id,
            len(tenant.policy.index),
            "on" if recording.enabled else "off",
        )

    return create_app(
        registry,
        stt=WhisperSTT(s.whisper_model, s.whisper_device),
        tts=VoicevoxTTS(s.voicevox_url, s.voicevox_speaker),
        call_idle_timeout_s=s.call_idle_timeout_s,
        handoff_ttl_s=s.handoff_ttl_s,
        outbox=Outbox(ttl_s=s.record_ttl_s, lease_s=s.record_lease_s),
        deployment=manifest,
        recording_store=store,
        admin_token_sha256=s.admin_token_sha256 or None,
        recorder_src=s.recorder_src,
    )


app = build_app()
