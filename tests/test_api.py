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


# ------------------------------------------------------------------ audio decoding


def _wav(rate=24000, seconds=1.0, channels=1, width=2):
    import io
    import wave

    import numpy as np

    t = np.arange(int(rate * seconds)) / rate
    tone = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    if channels == 2:
        tone = np.repeat(tone, 2)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(tone.tobytes())
    return buf.getvalue()


@pytest.mark.parametrize("rate,channels", [(24000, 1), (8000, 1), (16000, 2)])
def test_decode_wav_resamples_to_16k_mono(rate, channels):
    from callcenter.adapters import decode_wav

    pcm = decode_wav(_wav(rate=rate, channels=channels))
    assert pcm.dtype.name == "float32" and pcm.ndim == 1
    assert abs(len(pcm) - 16000) <= 1
    assert 0.45 < float(abs(pcm).max()) <= 0.51


def test_decode_wav_rejects_non_wav():
    from callcenter.adapters import decode_wav

    assert decode_wav(b"not audio") is None
    assert decode_wav(b"") is None


def test_transcription_error_returns_422_and_keeps_call(embedder, llm, faq):
    from callcenter.adapters import TranscriptionError

    class BrokenSTT:
        def transcribe(self, audio, language="ja"):
            raise TranscriptionError("decoder failed")

    registry = TenantRegistry(embedder, llm)
    registry.register("demo-clinic", "Demo", faq)
    c = TestClient(create_app(registry, BrokenSTT(), FakeTTS()))
    cid = _call(c)
    r = c.post(
        f"/v1/tenants/demo-clinic/calls/{cid}/audio",
        files={"file": ("u.wav", b"RIFF....", "audio/wav")},
    )
    assert r.status_code == 422 and "speech recognition failed" in r.json()["detail"]
    # the call is still usable afterwards
    ok = c.post(f"/v1/tenants/demo-clinic/calls/{cid}/text", json={"text": "診療時間"})
    assert ok.status_code == 200
