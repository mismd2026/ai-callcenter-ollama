import base64
from urllib.parse import unquote

import pytest
from conftest import FakeSTT, FakeTTS
from fastapi.testclient import TestClient

from callcenter.api import create_app
from callcenter.tenants import TenantRegistry


@pytest.fixture
def client(embedder, llm, faq):
    registry = TenantRegistry(embedder, llm)
    registry.register("demo-clinic", "Demo", faq)
    registry.register("other-co", "Other", [])
    return TestClient(create_app(registry, FakeSTT(), FakeTTS()))


def _call(client, tenant="demo-clinic"):
    r = client.post(f"/v1/tenants/{tenant}/calls")
    assert r.status_code == 201
    return r.json()["call_id"]


def test_text_turn_returns_layer_and_audio(client):
    cid = _call(client)
    r = client.post(
        f"/v1/tenants/demo-clinic/calls/{cid}/text",
        json={"text": "駐車場はありますか", "include_audio": True},
    )
    body = r.json()
    assert r.status_code == 200
    assert body["layer"] == 1
    assert body["sources"] == ["access-01"]
    assert base64.b64decode(body["audio_wav_base64"]).startswith(b"RIFF")


def test_audio_turn_round_trip(client):
    cid = _call(client)
    r = client.post(
        f"/v1/tenants/demo-clinic/calls/{cid}/audio",
        files={"file": ("u.wav", "解約したいです".encode(), "audio/wav")},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers["X-Escalation-Layer"] == "3"
    assert unquote(r.headers["X-Transcript"]) == "解約したいです"


def test_tenants_are_isolated(client):
    cid = _call(client, "demo-clinic")
    # same call id under another tenant must not resolve
    r = client.post(f"/v1/tenants/other-co/calls/{cid}/text", json={"text": "診療時間"})
    assert r.status_code == 404
    # other tenant has no FAQ -> never sees demo-clinic answers
    cid2 = _call(client, "other-co")
    r = client.post(f"/v1/tenants/other-co/calls/{cid2}/text", json={"text": "診療時間"})
    assert r.json()["layer"] == 3


def test_unknown_tenant_and_validation(client):
    assert client.post("/v1/tenants/nope/calls").status_code == 404
    cid = _call(client)
    r = client.post(f"/v1/tenants/demo-clinic/calls/{cid}/text", json={"text": ""})
    assert r.status_code == 422


def test_metrics_count_turns_by_layer(client):
    cid = _call(client)
    client.post(f"/v1/tenants/demo-clinic/calls/{cid}/text", json={"text": "診療時間"})
    text = client.get("/metrics").text
    assert 'callcenter_turns_total{tenant="demo-clinic",layer="1"} 1' in text
