from conftest import FakeLLM, FakeSTT, FakeTTS
from fastapi.testclient import TestClient

from callcenter.api import create_app
from callcenter.handoff import HandoffSummarizer, mask_numbers
from callcenter.tenants import TenantRegistry


class FailingLLM:
    def chat(self, messages, *, temperature: float = 0.2) -> str:
        raise RuntimeError("ollama down")


def _client(embedder, faq, llm, summarizer=None):
    registry = TenantRegistry(embedder, llm)
    registry.register("demo-clinic", "Demo", faq)
    return TestClient(create_app(registry, FakeSTT(), FakeTTS(), summarizer))


def _turn(client, cid, text):
    return client.post(f"/v1/tenants/demo-clinic/calls/{cid}/text", json={"text": text}).json()


# ------------------------------------------------------------------- summarizer unit


def test_summary_uses_llm_and_keeps_transcript():
    llm = FakeLLM("駐車場の料金について確認したうえで、解約を希望されています。")
    s = HandoffSummarizer(llm).summarize(
        [("駐車場はありますか", "裏手にございます。"), ("解約したい", "おつなぎします。")],
        "hard rule matched: 解約",
        ["access-01", "access-01"],
    )
    assert s.generated_by == "llm"
    assert s.summary == llm.reply
    assert s.last_utterance == "解約したい"
    assert [t["role"] for t in s.transcript] == ["caller", "ai", "caller", "ai"]
    assert s.faq_sources == ["access-01"]  # de-duplicated
    assert "解約したい" in llm.messages[0][1]["content"]


def test_summary_falls_back_when_llm_fails():
    s = HandoffSummarizer(FailingLLM()).summarize([("人と話したい", "おつなぎします。")], "r")
    assert s.generated_by == "fallback"
    assert "人と話したい" in s.summary


def test_long_numbers_are_masked_everywhere():
    llm = FakeLLM("カード 4111 1111 1111 1111 の件です。")
    s = HandoffSummarizer(llm).summarize([("カードは4111-1111-1111-1111です", "…")], "r")
    for text in (s.summary, s.last_utterance, s.transcript[0]["text"]):
        assert "4111" not in text
    assert mask_numbers("予約は3日の15時") == "予約は3日の15時"  # short numbers untouched


# ------------------------------------------------------------------- API


def test_handoff_brief_available_after_escalation(embedder, faq, llm):
    client = _client(embedder, faq, llm)
    cid = client.post("/v1/tenants/demo-clinic/calls").json()["call_id"]
    assert _turn(client, cid, "駐車場はありますか")["handoff_url"] is None

    body = _turn(client, cid, "解約したいです")
    assert body["layer"] == 3
    url = body["handoff_url"]
    assert url == f"/v1/tenants/demo-clinic/calls/{cid}/handoff"

    brief = client.get(url).json()
    assert brief["status"] == "ready"
    assert brief["generated_by"] == "llm"
    assert brief["reason"].startswith("hard rule")
    assert brief["last_utterance"] == "解約したいです"
    assert brief["faq_sources"] == ["access-01"]
    assert brief["transcript"][0] == {"role": "caller", "text": "駐車場はありますか"}


def test_handoff_404_before_escalation_and_across_tenants(embedder, faq, llm):
    registry = TenantRegistry(embedder, llm)
    registry.register("demo-clinic", "Demo", faq)
    registry.register("other-co", "Other", [])
    client = TestClient(create_app(registry, FakeSTT(), FakeTTS()))
    cid = client.post("/v1/tenants/demo-clinic/calls").json()["call_id"]
    _turn(client, cid, "診療時間を教えてください")
    assert client.get(f"/v1/tenants/demo-clinic/calls/{cid}/handoff").status_code == 404

    _turn(client, cid, "人と話したい")
    assert client.get(f"/v1/tenants/demo-clinic/calls/{cid}/handoff").status_code == 200
    assert client.get(f"/v1/tenants/other-co/calls/{cid}/handoff").status_code == 404


def test_audio_turn_returns_handoff_header(embedder, faq, llm):
    client = _client(embedder, faq, llm)
    cid = client.post("/v1/tenants/demo-clinic/calls").json()["call_id"]
    r = client.post(
        f"/v1/tenants/demo-clinic/calls/{cid}/audio",
        files={"file": ("u.wav", "オペレーターに代わってください".encode(), "audio/wav")},
    )
    assert r.headers["X-Escalation-Layer"] == "3"
    assert client.get(r.headers["X-Handoff-Url"]).json()["status"] == "ready"


def test_summary_failure_never_breaks_the_handoff(embedder, faq, llm):
    client = _client(embedder, faq, llm, summarizer=HandoffSummarizer(FailingLLM()))
    cid = client.post("/v1/tenants/demo-clinic/calls").json()["call_id"]
    body = _turn(client, cid, "解約したいです")
    assert body["layer"] == 3 and body["reply"]
    assert client.get(body["handoff_url"]).json()["generated_by"] == "fallback"
