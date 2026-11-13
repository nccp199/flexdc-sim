"""Regression tests for the portable Harness protocol."""
from hashlib import sha256
from unittest.mock import patch

from peacsim.gpt_dispatch.decisions import dumps
from peacsim.gpt_dispatch.observation import observe, policy_contract
from peacsim.rl.rewards import reward_spec
from peacsim.scenarios import load_scenario

from experiments.gpt6_sol_flexdc_dynamic.harness import run as harness
from experiments.gpt6_sol_flexdc_dynamic.harness.online import OnlinePayload
from experiments.gpt6_sol_flexdc_dynamic.harness.policy import HarnessPolicy


def test_original_guidance_and_hourly_defaults():
    guide, objective = harness.guidance(100, 90)
    assert sha256(guide.encode()).hexdigest() == "cb7039fb1f95bfc37eefa572d92fd92cb13e66cc7b9ddd4541b10995f416315f"
    assert sha256(objective.encode()).hexdigest() == "d5d1cb7e875e80300565ddbd7cb79233192aa3a84e1fe266814dbd3729e0b20e"
    cli = harness.parser()
    args = cli.parse_args(["--mode", "mock", "--output", "unused"])
    p = harness.resolve(args, cli, load_scenario(harness.SCENARIO))
    assert p["physical_steps"] == 3600
    assert p["wall_budget_seconds"] == 43200
    assert p["max_model_attempts"] == 10800
    assert p["reported_total_tokens_stop_threshold"] == 576_000_000
    assert p["completion_target"] is None
    guide, objective = harness.guidance(3600, None)
    assert "评价窗口为 3600 秒" in guide
    assert "原生截止为 t=100" in guide
    assert "90 项" not in objective


def test_rotation_preserves_feedback_and_global_budgets():
    def render(request, cohort):
        return dict(request)

    render.objective = "Deliver arrived tasks."
    protocol = dict(evaluation_cohort={"job_ids": [3]}, completion_target=None,
                    wall_budget_seconds=100, reported_total_tokens_stop_threshold=1000,
                    max_model_attempts=2, thread_steps=1)
    h = HarnessPolicy(protocol, render, model="test", effort="low", schema={})
    reply = dict(raw_output=dumps({"response": {"action": {}}}), usage={"totalTokens": 10})
    with patch.object(h, "_turn", wraps=h._turn), patch(
            "peacsim.gpt_dispatch.codex.CodexPolicy.decide", return_value=reply), patch.object(h, "close") as close:
        first = h.decide({"t": 0}, timeout_seconds=2)
        feedback = {"t": 1, "last": {"executed": "feedback"}, "operator_notices": ["Current event."]}
        second = h.decide(feedback, timeout_seconds=2)
        third = h.decide({"t": 2}, timeout_seconds=2)
    close.assert_called_once()
    assert "operator_notices" not in first["model_input"]
    assert second["model_input"]["operator_notices"][0] == render.objective
    assert second["model_input"]["operator_notices"][-1] == "Current event."
    assert second["model_input"]["last"] == feedback["last"]
    assert h.calls == 2 and h.tokens == 20
    assert third["retryable"] is False

def test_online_renderer_starts_without_experiment_archives(tmp_path):
    scenario = load_scenario(harness.SCENARIO)
    config, weights = harness._native_inputs(scenario, None, None)
    env = harness.FlexDCEnergyEnv(scenario, it_policy_config=config, weights=weights,
                                  scheduling_seed=101, output_root=tmp_path / "episode")
    reward = reward_spec(scenario, harness.load_reward_config(harness.REWARD_CONFIG))
    request = dict(observation=observe(env, scenario), contract=policy_contract(env, scenario, reward),
                   recent_decisions=[], validation_feedback=None, experiment_steps_remaining=3600)
    payload = OnlinePayload("Current objective.")(request, frozenset(harness.COHORT))
    assert payload["operator_notices"] == ["Current objective."]
    assert payload["time_left"] == 3600
    assert payload["last"] is None
