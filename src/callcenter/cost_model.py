"""Capacity and cost model for sizing a tenant's deployment.

Answers the sales question "how many GPUs does a 30-seat call center need,
and what is the cost per call?" from a few measurable inputs.

Erlang C is used for the human-operator side (Layer 3); the AI side is modelled
as a pool of GPU slots, each handling one concurrent call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def erlang_c(traffic_erlangs: float, agents: int) -> float:
    """Probability that a call has to wait (Erlang C)."""
    if agents <= traffic_erlangs:
        return 1.0
    a, n = traffic_erlangs, agents
    summation = sum(a**k / math.factorial(k) for k in range(n))
    top = a**n / math.factorial(n) * n / (n - a)
    return top / (summation + top)


def agents_for_service_level(
    traffic_erlangs: float, aht_sec: float, target_sl: float = 0.8, answer_within_sec: float = 20
) -> int:
    """Minimum agents so that `target_sl` of calls are answered within the threshold."""
    n = max(1, math.ceil(traffic_erlangs))
    while True:
        pw = erlang_c(traffic_erlangs, n)
        sl = 1 - pw * math.exp(-(n - traffic_erlangs) * answer_within_sec / aht_sec)
        if sl >= target_sl:
            return n
        n += 1


@dataclass(frozen=True)
class Workload:
    peak_calls_per_hour: float
    avg_handle_time_sec: float = 240
    ai_resolution_rate: float = 0.6  # share of calls closed in Layer 1/2
    human_aht_sec: float = 360


@dataclass(frozen=True)
class Infra:
    concurrent_calls_per_gpu: int = 6  # measured: whisper-medium + 7B q4 + VOICEVOX
    gpu_monthly_cost_jpy: int = 120_000
    operator_monthly_cost_jpy: int = 350_000
    business_days_per_month: int = 21
    calls_per_day_ratio_to_peak: float = 7.0  # daily volume ≈ peak hour × ratio


@dataclass(frozen=True)
class Estimate:
    gpus: int
    operators: int
    monthly_cost_jpy: int
    monthly_calls: int
    cost_per_call_jpy: float


def estimate(w: Workload, infra: Infra | None = None) -> Estimate:
    infra = infra or Infra()
    ai_erlangs = w.peak_calls_per_hour * w.avg_handle_time_sec / 3600
    # 30% headroom so a burst does not drop calls on the AI side
    gpus = max(1, math.ceil(ai_erlangs * 1.3 / infra.concurrent_calls_per_gpu))

    human_calls = w.peak_calls_per_hour * (1 - w.ai_resolution_rate)
    human_erlangs = human_calls * w.human_aht_sec / 3600
    operators = agents_for_service_level(human_erlangs, w.human_aht_sec) if human_erlangs else 0

    monthly = gpus * infra.gpu_monthly_cost_jpy + operators * infra.operator_monthly_cost_jpy
    calls = round(
        w.peak_calls_per_hour * infra.calls_per_day_ratio_to_peak * infra.business_days_per_month
    )
    return Estimate(gpus, operators, monthly, calls, round(monthly / max(calls, 1), 1))
