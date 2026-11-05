import json

import pytest
from conftest import FakeSTT, FakeTTS
from fastapi.testclient import TestClient

from callcenter.api import create_app
from callcenter.records import Outbox, RecordingConfig, hash_token, new_token
from callcenter.tenants import TenantRegistry
from recorder import DirectoryStorage, Recorder, storage_from_url

TOKEN = "s3cret-recorder-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BASE = "/v1/tenants/demo-clinic"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def client(embedder, faq, llm, clock):
    registry = TenantRegistry(embedder, llm)
    on = RecordingConfig(enabled=True, token_sha256=hash_token(TOKEN))
    registry.register("demo-clinic", "Demo", faq, on)
    registry.register("no-rec", "NoRec", faq)  # recording off
    outbox = Outbox(ttl_s=3600, lease_s=60, clock=clock)
    app = create_app(
        registry, FakeSTT(), FakeTTS(), call_idle_timeout_s=60, clock=clock, outbox=outbox
    )
    return TestClient(app)


def _call(client, texts, tenant="demo-clinic", end=True):
    base = f"/v1/tenants/{tenant}/calls"
    cid = client.post(base).json()["call_id"]
    for t in texts:
        client.post(f"{base}/{cid}/text", json={"text": t})
    if end:
        client.delete(f"{base}/{cid}")
    return cid


def _pull(client, **params):
    r = client.get(f"{BASE}/records", headers=AUTH, params=params)
    assert r.status_code == 200
    return r.json()["records"]


# ------------------------------------------------------------------ outbox / token


def test_token_verification():
    tok, digest = new_token()
    cfg = RecordingConfig(enabled=True, token_sha256=digest)
    assert cfg.verify(f"Bearer {tok}")
    assert not cfg.verify("Bearer wrong")
    assert not cfg.verify(tok)  # scheme required
    assert not RecordingConfig(enabled=False, token_sha256=digest).verify(f"Bearer {tok}")


def test_outbox_lease_redelivers_unacknowledged(clock):
    box = Outbox(lease_s=60, clock=clock)
    box.put("t", {"record_id": "a"})
    assert [r["delivery_attempt"] for r in box.claim("t", 10)] == [1]
    assert box.claim("t", 10) == []  # leased
    clock.now += 61
    assert [r["delivery_attempt"] for r in box.claim("t", 10)] == [2]
    assert box.ack("t", "a") and not box.ack("t", "a")
    assert box.pending("t") == 0 and box.delivered["t"] == 1


def test_outbox_drops_uncollected_records(clock):
    box = Outbox(ttl_s=100, max_per_tenant=2, clock=clock)
    for rid in "abc":
        box.put("t", {"record_id": rid})
        clock.now += 1
    assert box.pending("t") == 2 and box.dropped["t"] == 1  # oldest evicted
    clock.now += 200
    box.sweep()
    assert box.pending("t") == 0 and box.dropped["t"] == 3


# ------------------------------------------------------------------ API


def test_finished_call_is_queued_with_turns_and_handoff(client):
    cid = _call(client, ["駐車場はありますか", "カード番号は4111111111111111です", "解約したい"])
    (rec,) = _pull(client)
    assert rec["record_id"] == cid and rec["end_reason"] == "ended"
    assert rec["final_layer"] == 3
    assert len(rec["turns"]) == 3 and rec["turns"][-1]["layer"] == 3
    assert rec["turns"][0]["sources"] == ["access-01"]
    assert "4111" not in json.dumps(rec, ensure_ascii=False)  # masked
    assert rec["handoff"]["reason"].startswith("hard rule")
    assert rec["started_at"] and rec["ended_at"]


def test_ack_deletes_record_from_memory(client):
    cid = _call(client, ["診療時間を教えてください"])
    _pull(client)
    assert client.post(f"{BASE}/records/{cid}/ack", headers=AUTH).status_code == 204
    assert client.post(f"{BASE}/records/{cid}/ack", headers=AUTH).status_code == 404
    assert 'callcenter_records_pending{tenant="demo-clinic"} 0' in client.get("/metrics").text


def test_idle_calls_are_recorded_too(client, clock):
    _call(client, ["駐車場はありますか"], end=False)
    clock.now += 61
    client.get("/healthz")  # any request runs the sweep
    (rec,) = _pull(client)
    assert rec["end_reason"] == "idle"


def test_no_records_when_recording_disabled(client):
    _call(client, ["駐車場はありますか"], tenant="no-rec")
    r = client.get("/v1/tenants/no-rec/records", headers=AUTH)
    assert r.status_code == 401


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}])
def test_records_require_valid_token(client, headers):
    _call(client, ["駐車場はありますか"])
    assert client.get(f"{BASE}/records", headers=headers).status_code == 401
    assert client.get("/v1/tenants/unknown/records", headers=AUTH).status_code == 401


# ------------------------------------------------------------------ recorder app


def test_recorder_stores_and_acknowledges(client, tmp_path):
    client.headers.update(AUTH)
    ids = [_call(client, ["駐車場はありますか"]) for _ in range(3)]
    rec = Recorder(client, "demo-clinic", DirectoryStorage(tmp_path), batch=2)
    assert rec.run_once() == 2 and rec.run_once() == 1 and rec.run_once() == 0
    files = sorted(p.stem for p in tmp_path.rglob("*.json"))
    assert files == sorted(ids)
    saved = json.loads(next(tmp_path.rglob(f"{ids[0]}.json")).read_text())
    assert saved["turns"][0]["caller"] == "駐車場はありますか"
    assert "delivery_attempt" not in saved
    assert not list(tmp_path.rglob("*.tmp"))


def test_recorder_does_not_ack_when_storage_fails(client, tmp_path, clock):
    client.headers.update(AUTH)
    _call(client, ["駐車場はありますか"])

    class Broken:
        def save(self, record):
            raise OSError("disk full")

    assert Recorder(client, "demo-clinic", Broken()).run_once() == 0
    clock.now += 61  # lease expires -> offered again, nothing lost
    assert Recorder(client, "demo-clinic", DirectoryStorage(tmp_path)).run_once() == 1


def test_directory_storage_is_idempotent(tmp_path):
    s = storage_from_url(f"dir://{tmp_path}")
    rec = {"record_id": "x", "ended_at": "2026-11-05T00:00:00+00:00"}
    assert s.save(rec) is True and s.save(rec) is False
    assert (tmp_path / "2026-11-05" / "x.json").exists()
