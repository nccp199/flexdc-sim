"""Run an independent GPT or mock scheduler at native FlexDC physical cadence."""
import argparse
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

from peacsim.envs import FlexDCEnergyEnv
from peacsim.gpt_dispatch.codex import CodexPolicy
from peacsim.gpt_dispatch.decisions import INSTRUCTIONS, MockPolicy, decision_schema, dumps
from peacsim.gpt_dispatch.responses import ResponsesPolicy
from peacsim.gpt_dispatch.runner import run_episode
from peacsim.parsing.policy_config_reader import PolicyConfigReader
from peacsim.rl.rewards import load_reward_config
from peacsim.scenarios import load_scenario


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "scripts/validation/configs"


def _native_inputs(scenario, policy_path, weights_path):
    if policy_path is None and weights_path is None and scenario.scenario_id == "micro_timing":
        return SimpleNamespace(runtime_policy_name="AQA", P=6500.0, R=100.0,
                               policy_parameters={"node_count_control": False}), {0: 1.0}
    policy = PolicyConfigReader(policy_path or NATIVE / "policy.ini")
    policy.policy_parameters.pop("weights_path", None)
    values = pd.read_csv(weights_path or NATIVE / "original_weights.csv").iloc[:, 1]
    if len(values) != scenario.job_config.job_type_count:
        raise ValueError("Supply one AQA initialization weight per configured task type, in native order")
    return policy, {i: float(value) for i, value in enumerate(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "codex", "responses"), required=True)
    parser.add_argument("--scenario", type=Path, default=ROOT / "configs/scenarios/development_midday.json")
    parser.add_argument("--seed", "--scheduling-seed", type=int, default=20, help="Native scheduling RNG seed")
    parser.add_argument("--steps", type=int, default=12, help="Maximum physical steps (default 12)")
    parser.add_argument("--model", help="Defaults to gpt-6.1-sol for Codex or gpt-5.4-mini for Responses")
    parser.add_argument("--reasoning-effort", help="Model-specific effort (Codex and Responses default to low)")
    parser.add_argument("--max-output-tokens", type=int, default=2048, help="Responses API output token limit")
    parser.add_argument("--timeout", type=float, default=90.0, help="Per-step inference budget in seconds")
    parser.add_argument("--retries", type=int, default=1, help="Additional attempts within the step budget")
    parser.add_argument("--history-steps", type=int, default=2, help="Recent feedback included in each request")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-home", type=Path, help="Use an already authenticated Codex home without MCP servers")
    parser.add_argument("--codex-instructions", type=Path, help="Append a UTF-8 operating guide to the Codex instructions")
    parser.add_argument("--require-same-codex-thread", action="store_true",
                        help="Stop if an in-flight Codex thread is lost")
    parser.add_argument("--policy", type=Path, help="Native AQA initialization policy INI")
    parser.add_argument("--weights", type=Path, help="Native AQA initialization weight CSV")
    parser.add_argument("--reward-config", type=Path, default=ROOT / "configs/rl/reward_diagnostic_v1.json")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if args.steps < 1 or args.history_steps < 1 or args.retries < 0:
        parser.error("steps and history-steps must be positive, retries must be nonnegative")
    if args.max_output_tokens < 1:
        parser.error("max-output-tokens must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive and finite")
    if args.mode != "codex" and (args.codex_instructions or args.require_same_codex_thread):
        parser.error("Codex instruction and thread options require --mode codex")

    backend = None
    try:
        instructions = INSTRUCTIONS
        if args.codex_instructions:
            guide = args.codex_instructions.read_text(encoding="utf-8").strip()
            if not guide:
                raise ValueError("Codex operating guide is empty")
            instructions += "\n" + guide
        scenario = load_scenario(args.scenario)
        policy, weights = _native_inputs(scenario, args.policy, args.weights)
        reward_config = load_reward_config(args.reward_config)
        env = FlexDCEnergyEnv(scenario, it_policy_config=policy, weights=weights,
                              scheduling_seed=args.seed, output_root=args.output)
        schema = decision_schema(env.get_action_spec())
        effort = args.reasoning_effort or "low"
        if args.mode == "codex":
            backend = CodexPolicy(model=args.model or "gpt-6.1-sol", effort=effort,
                                  schema=schema, executable=args.codex_bin, codex_home=args.codex_home,
                                  instructions=instructions, require_same_thread=args.require_same_codex_thread)
        elif args.mode == "responses":
            backend = ResponsesPolicy(model=args.model or "gpt-5.4-mini", effort=effort,
                                      schema=schema, max_output_tokens=args.max_output_tokens)
        else:
            backend = MockPolicy()
        summary = run_episode(env, scenario, backend, steps=args.steps, timeout_seconds=args.timeout,
                              retries=args.retries, history_steps=args.history_steps, reward_config=reward_config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Dispatch failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if isinstance(backend, CodexPolicy):
            backend.close()
    print(dumps({key: summary[key] for key in ("status", "steps_completed", "simulation_seconds", "method", "model")}))
    print(f"Logs: {env.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
