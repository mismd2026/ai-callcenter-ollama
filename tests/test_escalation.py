import pytest

from callcenter.escalation import ESCALATE_SENTINEL, Layer


def test_layer1_reads_faq_answer_verbatim_without_llm(policy, state, llm):
    d = policy.decide("診療時間を教えてください", state)
    assert d.layer is Layer.FAQ_DIRECT
    assert d.reply.startswith("診療時間は平日9時から18時")
    assert d.sources == ["hours-01"]
    assert llm.messages == []  # deterministic path, no generation


def test_layer2_grounds_llm_on_retrieved_passages(policy, state, llm):
    d = policy.decide("予約して行くとき駐車場は使えますか", state)
    assert d.layer is Layer.LLM_GROUNDED
    assert d.reply == llm.reply
    prompt = llm.messages[0][1]["content"]
    assert "access-01" in prompt and "reserve-" in prompt
    assert ESCALATE_SENTINEL in llm.messages[0][0]["content"]


def test_layer2_llm_refusal_hands_off(policy, state, llm):
    llm.reply = ESCALATE_SENTINEL
    d = policy.decide("予約して行くとき駐車場は使えますか", state)
    assert d.layer is Layer.HUMAN
    assert "declined" in d.reason


def test_unknown_topic_hands_off(policy, state, llm):
    d = policy.decide("ペットを連れて行ってもいいですか", state)
    assert d.layer is Layer.HUMAN
    assert llm.messages == []


@pytest.mark.parametrize(
    "utterance",
    ["人と話したいです", "オペレーターに代わってください", "解約したい", "クレームを言いたい"],
)
def test_hard_rules_override_confident_retrieval(policy, state, utterance):
    d = policy.decide(f"{utterance}。診療時間は？", state)
    assert d.layer is Layer.HUMAN
    assert d.reason.startswith("hard rule")


def test_caller_repeating_question_is_handed_off(policy, state):
    for _ in range(2):
        assert policy.decide("診療時間は？", state).layer is Layer.FAQ_DIRECT
    assert policy.decide("診療時間は？", state).layer is Layer.HUMAN


def test_turn_budget_exhausted_hands_off(policy, state):
    policy.max_ai_turns = 2
    policy.decide("予約して行くとき駐車場は使えますか", state)
    policy.decide("駐車場と予約、もう一度確認させて", state)
    d = policy.decide("診断書の予約もできますか", state)
    assert d.layer is Layer.HUMAN
    assert "budget" in d.reason
