"""Three-layer response / escalation policy.

    Layer 1  FAQ direct answer   - near-exact FAQ match, answer read out verbatim.
                                   Deterministic, no generation -> zero hallucination risk.
    Layer 2  Grounded LLM answer - LLM answers ONLY from retrieved FAQ passages.
                                   It must reply with the sentinel if the passages
                                   do not cover the question.
    Layer 3  Human operator      - hard rules (complaints, cancellation, explicit
                                   request for a person), low retrieval confidence,
                                   LLM refusal, or the caller going round in circles.

Hard rules are checked first: a caller who asks for a human is never kept in the
AI loop, regardless of how confident retrieval is.

Follow-up questions ("それは無料ですか") carry no topic of their own, so retrieval
also runs on the previous caller utterance + the current one, and the LLM sees
the last few turns. The contextual search only feeds Layer 2: Layer 1 still
requires the current utterance alone to match, so an old topic is never read
out verbatim in reply to a new question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum

from .adapters import ChatModel
from .rag import FaqIndex, Hit

ESCALATE_SENTINEL = "<<ESCALATE>>"

# Phrases that always route to a human. Kept as data so each tenant can extend it.
DEFAULT_HANDOFF_PATTERNS: tuple[str, ...] = (
    r"(人|オペレーター|担当者|責任者).*(話|代わ|つな|繋)",
    r"解約",
    r"(苦情|クレーム)",
    r"(弁護士|訴え|消費者センター)",
    r"(返金|返品).*(したい|して)",
)

SYSTEM_PROMPT = """あなたはコールセンターの音声応対AIです。
- 以下の「参考情報」に書かれている内容だけを根拠に答えてください。
- 電話口で読み上げやすい丁寧な日本語で、2〜3文にまとめてください。
- これまでの会話の流れを踏まえ、「それ」「その場合」などが何を指すかを判断してください。
  ただし回答の根拠にしてよいのは参考情報だけです。
- 参考情報で答えられない場合、推測せずに必ず {sentinel} とだけ出力してください。
- 個人情報（カード番号、パスワード等）を聞き出してはいけません。
"""


class Layer(IntEnum):
    FAQ_DIRECT = 1
    LLM_GROUNDED = 2
    HUMAN = 3


@dataclass
class Decision:
    layer: Layer
    reply: str
    reason: str
    sources: list[str] = field(default_factory=list)


@dataclass
class ConversationState:
    ai_turns_without_resolution: int = 0
    recent_queries: list[str] = field(default_factory=list)
    # (caller utterance, AI reply) pairs, oldest first. Kept in memory only.
    history: list[tuple[str, str]] = field(default_factory=list)

    @property
    def previous_utterance(self) -> str | None:
        return self.history[-1][0] if self.history else None


@dataclass
class EscalationPolicy:
    index: FaqIndex
    llm: ChatModel
    direct_threshold: float = 0.86
    context_threshold: float = 0.70
    max_ai_turns: int = 3
    max_history_turns: int = 3
    handoff_patterns: tuple[str, ...] = DEFAULT_HANDOFF_PATTERNS

    handoff_message: str = "担当のオペレーターにおつなぎします。少々お待ちください。"

    def decide(self, utterance: str, state: ConversationState) -> Decision:
        text = utterance.strip()
        decision = self._decide(text, state)
        state.history.append((text, decision.reply))
        if self.max_history_turns >= 0:
            del state.history[: max(0, len(state.history) - self.max_history_turns)]
        return decision

    def _decide(self, text: str, state: ConversationState) -> Decision:

        # --- Layer 3 hard rules -----------------------------------------------------
        for pat in self.handoff_patterns:
            if re.search(pat, text):
                return self._handoff(f"hard rule matched: {pat}")
        if self._is_repeating(text, state):
            return self._handoff("caller repeated the same question")
        if state.ai_turns_without_resolution >= self.max_ai_turns:
            return self._handoff("AI turn budget exhausted")

        previous = state.previous_utterance
        state.recent_queries.append(text)
        hits = self.index.search(text, k=3)
        best = hits[0].score if hits else 0.0

        # --- Layer 1 (current utterance only) ---------------------------------------
        if best >= self.direct_threshold:
            state.ai_turns_without_resolution = 0
            return Decision(
                Layer.FAQ_DIRECT,
                hits[0].entry.answer,
                f"faq score {best:.2f}",
                [hits[0].entry.id],
            )

        # --- Layer 2 (current utterance + conversation context) ---------------------
        used_context = False
        if previous and self.max_history_turns > 0:
            contextual = self.index.search(f"{previous} {text}", k=3)
            merged = self._merge(hits, contextual)
            used_context = bool(merged) and merged[0].score > best
            hits = merged
        best = hits[0].score if hits else 0.0

        context = [h for h in hits if h.score >= self.context_threshold]
        if not context:
            state.ai_turns_without_resolution += 1
            return self._handoff(f"no relevant FAQ (best {best:.2f})")

        history = state.history[-self.max_history_turns :] if self.max_history_turns > 0 else []
        answer = self.llm.chat(self._grounded_messages(text, context, history))
        if ESCALATE_SENTINEL in answer or not answer:
            state.ai_turns_without_resolution += 1
            return self._handoff("LLM declined: question not covered by FAQ")

        state.ai_turns_without_resolution += 1  # resolved only when caller confirms
        return Decision(
            Layer.LLM_GROUNDED,
            answer,
            f"grounded on {len(context)} passage(s), best {best:.2f}"
            + (" (with conversation context)" if used_context else ""),
            [h.entry.id for h in context],
        )

    # ------------------------------------------------------------------ helpers
    def _handoff(self, reason: str) -> Decision:
        return Decision(Layer.HUMAN, self.handoff_message, reason)

    @staticmethod
    def _is_repeating(text: str, state: ConversationState) -> bool:
        norm = re.sub(r"\W", "", text)
        return sum(re.sub(r"\W", "", q) == norm for q in state.recent_queries[-3:]) >= 2

    @staticmethod
    def _merge(*hit_lists: list[Hit]) -> list[Hit]:
        """Union of hit lists, keeping each FAQ entry's best score."""
        best: dict[str, Hit] = {}
        for hits in hit_lists:
            for h in hits:
                if h.entry.id not in best or h.score > best[h.entry.id].score:
                    best[h.entry.id] = h
        return sorted(best.values(), key=lambda h: -h.score)

    @staticmethod
    def _grounded_messages(
        question: str, context: list[Hit], history: list[tuple[str, str]] | None = None
    ) -> list[dict[str, str]]:
        refs = "\n".join(
            f"[{h.entry.id}] Q: {h.entry.question}\nA: {h.entry.answer}" for h in context
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(sentinel=ESCALATE_SENTINEL)}]
        for caller, reply in history or []:
            messages.append({"role": "user", "content": f"お客様の発話: {caller}"})
            messages.append({"role": "assistant", "content": reply})
        messages.append(
            {"role": "user", "content": f"参考情報:\n{refs}\n\nお客様の発話: {question}"}
        )
        return messages
