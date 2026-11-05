import pytest
from conftest import FakeSTT, FakeTTS
from fastapi.testclient import TestClient

from callcenter.api import create_app
from callcenter.tenants import TenantRegistry

BASE = "/v1/tenants/demo-clinic/calls"


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
    registry.register("demo-clinic", "Demo", faq)
    registry.register("other-co", "Other", [])
    app = create_app(
        registry, FakeSTT(), FakeTTS(), call_idle_timeout_s=60, handoff_ttl_s=300, clock=clock
    )
    c = TestClient(app)
    c.registry = registry
    return c


def _start(client):
    return client.post(BASE).json()["call_id"]


def _say(client, cid, text):
    return client.post(f"{BASE}/{cid}/text", json={"text": text})


def test_end_call_discards_conversation(client):
    cid = _start(client)
    _say(client, cid, "駐車場はありますか")
    r = client.delete(f"{BASE}/{cid}")
    assert r.status_code == 200
    assert r.json() == {"call_id": cid, "turns": 1, "final_layer": 1, "handoff_pending": False}
    assert cid not in client.registry.get("demo-clinic").sessions
    assert _say(client, cid, "診療時間は？").status_code == 404
    assert client.delete(f"{BASE}/{cid}").status_code == 404


def test_end_call_keeps_brief_for_operator_until_acknowledged(client):
    cid = _start(client)
    _say(client, cid, "人と話したい")
    assert client.delete(f"{BASE}/{cid}").json()["handoff_pending"] is True
    assert client.get(f"{BASE}/{cid}/handoff").json()["status"] == "ready"

    assert client.delete(f"{BASE}/{cid}/handoff").status_code == 204
    assert client.get(f"{BASE}/{cid}/handoff").status_code == 404


def test_idle_calls_expire(client, clock):
    cid = _start(client)
    _say(client, cid, "駐車場はありますか")
    clock.now += 59
    assert _say(client, cid, "診療時間は？").status_code == 200  # activity resets the timer
    clock.now += 61
    assert _say(client, cid, "診療時間は？").status_code == 404
    assert cid not in client.registry.get("demo-clinic").sessions


def test_handoff_briefs_expire(client, clock):
    cid = _start(client)
    _say(client, cid, "解約したい")
    client.delete(f"{BASE}/{cid}")
    clock.now += 301
    assert client.get(f"{BASE}/{cid}/handoff").status_code == 404


def test_other_tenant_cannot_end_or_acknowledge(client):
    cid = _start(client)
    _say(client, cid, "解約したい")
    other = f"/v1/tenants/other-co/calls/{cid}"
    assert client.delete(other).status_code == 404
    assert client.delete(f"{other}/handoff").status_code == 404
    assert client.get(f"{BASE}/{cid}/handoff").json()["status"] == "ready"


def test_metrics_report_active_calls_and_closures(client, clock):
    a, b = _start(client), _start(client)
    _say(client, a, "人と話したい")
    client.delete(f"{BASE}/{a}")
    clock.now += 61
    text = client.get("/metrics").text
    assert 'callcenter_calls_closed_total{tenant="demo-clinic",why="ended"} 1' in text
    assert 'callcenter_calls_closed_total{tenant="demo-clinic",why="idle"} 1' in text
    assert "callcenter_active_calls 0" in text
    assert "callcenter_pending_handoffs 1" in text
    assert b not in client.registry.get("demo-clinic").sessions
