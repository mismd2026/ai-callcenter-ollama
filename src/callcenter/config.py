"""Runtime configuration, loaded from environment variables.

Every endpoint defaults to a service inside the cluster / compose network.
No third-party cloud AI endpoint is ever configured: all inference stays on-prem.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://ollama:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    voicevox_url: str = field(default_factory=lambda: _env("VOICEVOX_URL", "http://voicevox:50021"))
    voicevox_speaker: int = field(default_factory=lambda: int(_env("VOICEVOX_SPEAKER", "3")))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "medium"))
    whisper_device: str = field(default_factory=lambda: _env("WHISPER_DEVICE", "auto"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    )
    faq_dir: str = field(default_factory=lambda: _env("FAQ_DIR", "data/faq"))
    # Escalation thresholds (cosine similarity, 0..1)
    faq_direct_threshold: float = field(
        default_factory=lambda: float(_env("FAQ_DIRECT_THRESHOLD", "0.86"))
    )
    rag_context_threshold: float = field(
        default_factory=lambda: float(_env("RAG_CONTEXT_THRESHOLD", "0.70"))
    )
    max_ai_turns_without_resolution: int = field(
        default_factory=lambda: int(_env("MAX_AI_TURNS", "3"))
    )
    # Past turns given to the LLM / used for follow-up retrieval (0 disables).
    max_history_turns: int = field(default_factory=lambda: int(_env("MAX_HISTORY_TURNS", "3")))
    # In-memory retention of conversation data (seconds).
    call_idle_timeout_s: float = field(
        default_factory=lambda: float(_env("CALL_IDLE_TIMEOUT_S", "600"))
    )
    handoff_ttl_s: float = field(default_factory=lambda: float(_env("HANDOFF_TTL_S", "1800")))
    # Guard rail: refuse to start if an endpoint points outside the private network.
    allow_external_endpoints: bool = field(
        default_factory=lambda: _env("ALLOW_EXTERNAL_ENDPOINTS", "false").lower() == "true"
    )


_PRIVATE_SUFFIXES = (".svc", ".svc.cluster.local", ".local", ".internal", ".lan")


def is_private_host(url: str) -> bool:
    """True if the URL's host is an in-cluster name, localhost or a private IP."""
    host = urlsplit(url).hostname or ""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost" or "." not in host or host.endswith(_PRIVATE_SUFFIXES)
    return ip.is_private or ip.is_loopback


def assert_data_sovereignty(settings: Settings) -> None:
    """Fail fast if an AI endpoint looks like a public cloud service.

    This enforces the project's policy that caller audio and transcripts never
    leave the operator's own infrastructure.
    """
    if settings.allow_external_endpoints:
        return
    for name, url in (("OLLAMA_URL", settings.ollama_url), ("VOICEVOX_URL", settings.voicevox_url)):
        if not is_private_host(url):
            raise RuntimeError(
                f"{name}={url} looks external. Set ALLOW_EXTERNAL_ENDPOINTS=true to override."
            )
