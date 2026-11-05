"""Conversation records handed to the customer's own storage.

The call center never writes into customer storage and never holds credentials
for it. Instead a small recorder app installed on the customer's side
(see recorder/) connects to this API, pulls finished call records, stores them
wherever the customer chooses, and acknowledges each one. Only then is the
record deleted from memory here.

    call ends ──> Outbox (memory) ──pull──> customer recorder ──> customer storage
                         ^                        │
                         └─────── ack ────────────┘   -> deleted from memory

Delivery is at-least-once: a record that is pulled but not acknowledged within
`lease_s` is offered again, so the recorder must write idempotently (one file
per record_id). Records nobody collects are dropped after `ttl_s` rather than
kept indefinitely: holding data is the failure mode this design avoids.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RecordingConfig:
    """Per-tenant setting. Only the SHA-256 of the recorder token is stored."""

    enabled: bool = False
    token_sha256: str = ""

    @classmethod
    def from_dict(cls, data: dict | None) -> RecordingConfig:
        if not data:
            return cls()
        return cls(bool(data.get("enabled")), str(data.get("token_sha256", "")).lower())

    def verify(self, authorization: str | None) -> bool:
        if not (self.enabled and self.token_sha256 and authorization):
            return False
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return False
        return hmac.compare_digest(hash_token(token.strip()), self.token_sha256)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def new_token() -> tuple[str, str]:
    """Return (token for the recorder, hash for tenant.json)."""
    token = secrets.token_urlsafe(32)
    return token, hash_token(token)


@dataclass
class _Entry:
    record: dict
    created: float
    lease_until: float = 0.0
    attempts: int = 0


@dataclass
class Outbox:
    ttl_s: float = 86400.0
    lease_s: float = 60.0
    max_per_tenant: int = 1000
    clock: Callable[[], float] = time.monotonic
    _by_tenant: dict[str, dict[str, _Entry]] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    delivered: dict[str, int] = field(default_factory=dict)

    def put(self, tenant_id: str, record: dict) -> None:
        box = self._by_tenant.setdefault(tenant_id, {})
        while len(box) >= self.max_per_tenant:  # never grow without bound
            oldest = min(box, key=lambda k: box[k].created)
            del box[oldest]
            self.dropped[tenant_id] = self.dropped.get(tenant_id, 0) + 1
        box[record["record_id"]] = _Entry(record, self.clock())

    def claim(self, tenant_id: str, limit: int) -> list[dict]:
        now = self.clock()
        out: list[dict] = []
        box = self._by_tenant.get(tenant_id, {})
        for entry in sorted(box.values(), key=lambda e: e.created):
            if len(out) >= limit:
                break
            if entry.lease_until > now:
                continue  # handed to a recorder, waiting for its ack
            entry.lease_until = now + self.lease_s
            entry.attempts += 1
            out.append({**entry.record, "delivery_attempt": entry.attempts})
        return out

    def ack(self, tenant_id: str, record_id: str) -> bool:
        box = self._by_tenant.get(tenant_id, {})
        if box.pop(record_id, None) is None:
            return False
        self.delivered[tenant_id] = self.delivered.get(tenant_id, 0) + 1
        return True

    def sweep(self) -> None:
        now = self.clock()
        for tenant_id, box in self._by_tenant.items():
            for rid in [r for r, e in box.items() if now - e.created > self.ttl_s]:
                del box[rid]
                self.dropped[tenant_id] = self.dropped.get(tenant_id, 0) + 1

    def pending(self, tenant_id: str) -> int:
        return len(self._by_tenant.get(tenant_id, {}))

    def tenants(self) -> list[str]:
        return sorted(set(self._by_tenant) | set(self.dropped) | set(self.delivered))


if __name__ == "__main__":  # python -m callcenter.records
    tok, digest = new_token()
    print("recorder token (give to the customer, store nowhere else):")
    print(f"  {tok}")
    print('tenant.json:  "recording": {"enabled": true, "token_sha256": "' + digest + '"}')
