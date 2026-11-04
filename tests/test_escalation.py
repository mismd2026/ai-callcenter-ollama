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


# ------------------------------------------------------------- follow-up questions


def test_follow_up_without_topic_uses_previous_utterance(policy, state, llm):
    assert policy.decide("駐車場はありますか", state).layer is Layer.FAQ_DIRECT
    d = policy.decide("それは無料ですか", state)
    assert d.layer is Layer.LLM_GROUNDED
    assert d.sources == ["access-01"]
    assert "conversation context" in d.reason


def test_llm_receives_previous_turns_in_order(policy, state, llm):
    first = policy.decide("駐車場はありますか", state)
    policy.decide("それは無料ですか", state)
    msgs = llm.messages[0]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "駐車場はありますか" in msgs[1]["content"]
    assert msgs[2]["content"] == first.reply
    assert "それは無料ですか" in msgs[3]["content"] and "access-01" in msgs[3]["content"]


def test_previous_topic_is_never_read_out_verbatim(policy, state, llm):
    # Contextual retrieval matches access-01 perfectly, but the current utterance
    # alone does not, so the old FAQ answer must not be replayed as Layer 1.
    policy.decide("駐車場はありますか", state)
    d = policy.decide("それは無料ですか", state)
    assert d.layer is not Layer.FAQ_DIRECT


def test_history_is_capped(policy, state):
    policy.max_history_turns = 2
    for q in ["診療時間は？", "駐車場はありますか", "診断書はすぐもらえますか"]:
        policy.decide(q, state)
    assert [u for u, _ in state.history] == ["駐車場はありますか", "診断書はすぐもらえますか"]


def test_history_disabled_restores_single_turn_behaviour(policy, state, llm):
    policy.max_history_turns = 0
    policy.decide("駐車場はありますか", state)
    d = policy.decide("それは無料ですか", state)
    assert d.layer is Layer.HUMAN
    assert state.history == []
    assert llm.messages == []


def test_sessions_do_not_share_history(policy, llm):
    from callcenter.escalation import ConversationState

    a, b = ConversationState(), ConversationState()
    policy.decide("駐車場はありますか", a)
    d = policy.decide("それは無料ですか", b)
    assert d.layer is Layer.HUMAN
    assert b.history == [("それは無料ですか", policy.handoff_message)]
