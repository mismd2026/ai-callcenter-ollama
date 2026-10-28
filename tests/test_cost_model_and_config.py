import pytest

from callcenter.config import Settings, assert_data_sovereignty, is_private_host
from callcenter.cost_model import Workload, agents_for_service_level, erlang_c, estimate
from callcenter.tenants import TenantRegistry


def test_erlang_c_known_value():
    # Textbook example: 10 Erlangs, 12 agents -> P(wait) ≈ 0.4494
    assert erlang_c(10, 12) == pytest.approx(0.4494, abs=1e-3)


def test_service_level_needs_more_agents_than_load():
    assert agents_for_service_level(10, aht_sec=180) > 10


def test_estimate_scales_with_ai_resolution():
    low = estimate(Workload(peak_calls_per_hour=120, ai_resolution_rate=0.3))
    high = estimate(Workload(peak_calls_per_hour=120, ai_resolution_rate=0.8))
    assert high.operators < low.operators
    assert high.cost_per_call_jpy < low.cost_per_call_jpy
    assert high.gpus == low.gpus >= 1


@pytest.mark.parametrize(
    "url,private",
    [
        ("http://ollama:11434", True),
        ("http://ollama.ai.svc.cluster.local:11434", True),
        ("http://10.0.3.4:50021", True),
        ("http://localhost:11434", True),
        ("https://api.openai.com", False),
        ("http://8.8.8.8", False),
    ],
)
def test_private_host_detection(url, private):
    assert is_private_host(url) is private


def test_sovereignty_guard_blocks_cloud_endpoint():
    with pytest.raises(RuntimeError):
        assert_data_sovereignty(Settings(ollama_url="https://api.example-llm.com"))
    assert_data_sovereignty(
        Settings(ollama_url="https://api.example-llm.com", allow_external_endpoints=True)
    )


def test_tenant_id_validation(embedder, llm):
    with pytest.raises(ValueError):
        TenantRegistry(embedder, llm).register("../etc", "bad", [])
