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
from .rag import E5Embedder, load_faq
from .tenants import TenantRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("callcenter")


def build_app():
    s = Settings()
    assert_data_sovereignty(s)

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
        tenant = registry.register(tenant_dir.name, meta["name"], load_faq(tenant_dir))
        log.info("tenant %s loaded with %d FAQ entries", tenant.id, len(tenant.policy.index))

    return create_app(
        registry,
        stt=WhisperSTT(s.whisper_model, s.whisper_device),
        tts=VoicevoxTTS(s.voicevox_url, s.voicevox_speaker),
    )


app = build_app()
