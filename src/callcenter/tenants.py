"""Tenant registry.

In production tenants live in PostgreSQL (db/schema.sql) with row-level security,
so a bug in application code cannot leak one company's FAQ or call logs to another.
This module provides the in-process view used by the API layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .adapters import ChatModel
from .escalation import ConversationState, EscalationPolicy
from .rag import Embedder, FaqEntry, FaqIndex
from .records import RecordingConfig

TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,31}$")


@dataclass
class Tenant:
    id: str
    name: str
    policy: EscalationPolicy
    sessions: dict[str, ConversationState] = field(default_factory=dict)
    recording: RecordingConfig = field(default_factory=RecordingConfig)


class TenantRegistry:
    def __init__(self, embedder: Embedder, llm: ChatModel, **policy_kwargs) -> None:
        self._embedder = embedder
        self._llm = llm
        self._policy_kwargs = policy_kwargs
        self._tenants: dict[str, Tenant] = {}

    def register(
        self,
        tenant_id: str,
        name: str,
        faq: list[FaqEntry],
        recording: RecordingConfig | None = None,
    ) -> Tenant:
        if not TENANT_ID_RE.match(tenant_id):
            raise ValueError(f"invalid tenant id: {tenant_id!r}")
        index = FaqIndex(self._embedder)  # one index per tenant: no cross-tenant retrieval
        index.add(faq)
        policy = EscalationPolicy(index, self._llm, **self._policy_kwargs)
        tenant = Tenant(tenant_id, name, policy, recording=recording or RecordingConfig())
        self._tenants[tenant_id] = tenant
        return tenant

    def get(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    def __contains__(self, tenant_id: str) -> bool:
        return tenant_id in self._tenants
