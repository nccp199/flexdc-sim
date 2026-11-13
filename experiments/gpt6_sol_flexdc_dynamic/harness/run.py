"""Run FlexDC with an isolated Codex Harness or an explicit local test fixture."""
import argparse
from hashlib import sha256
import math
from pathlib import Path
import sys

from peacsim.envs import FlexDCEnergyEnv
from peacsim.gpt_dispatch.decisions import INSTRUCTIONS, MockPolicy, decision_schema, dumps
from peacsim.gpt_dispatch.observation import observe
from peacsim.gpt_dispatch.runner import run_episode
from peacsim.rl.rewards import load_reward_config
from peacsim.scenarios import load_scenario
from scripts.run_gpt_dispatch import _native_inputs

from .online import EVENT_CONTEXT, OnlinePayload
from .policy import HarnessPolicy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCENARIO = ROOT / "configs/scenarios/development_dynamic_100s.json"
REWARD_CONFIG = ROOT / "configs/rl/reward_diagnostic_v1.json"
BASE = HERE.parents[1] / "gpt6_sol_flexdc_intent/BASE_GUIDE.md"
MULTIOBJECTIVE = HERE.parent / "multiobjective"
SEED = 101
ORIGINAL_TARGET = 90
ORIGINAL_SECONDS = 100
CONDITION = "38b1e8106e182c45ef810856c835e7e27362a207e8233bc2df7944a04263e81e"
COHORT = (3, 11, 15, 25, 30, 37, 40, 43, 45, 48)


def read(path):
    return path.read_text(encoding="utf-8").strip()


def write(path, value):
    path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")


def guidance(seconds, target):
    guide = read(BASE).replace("评价窗口为 100 秒。", f"评价窗口为 {seconds} 秒。")
    objective = read(MULTIOBJECTIVE / "OBJECTIVE_A.md") + "\n\n" + read(MULTIOBJECTIVE / "OBJECTIVE_B2.md")
    if target is None:
        objective = (objective
            .replace("在本次 100 秒调度结束时，争取完成至少 90 项任务。", f"在本次 {seconds} 秒调度内，保障固定批次并尽量推进全场可交付任务。")
            .replace("这里的 90 项任务是当前种子 101 的实验目标，不应解释为其他场景的通用约束。\n", "")
            .replace("目标顺序和 90 项交付底线保持不变", "保持任务交付优先于购电成本")
            .replace("交付余量不必等到累计已完工达到 90 才判断。", "从开局起滚动判断交付余量。")
            .replace("达到 90 项后仍推进重要和接近完工的任务", "有交付余量时仍推进重要和接近完工的任务"))
    elif (seconds, target) != (ORIGINAL_SECONDS, ORIGINAL_TARGET):
        objective = objective.replace("100 秒调度", f"{seconds} 秒调度")
        objective = objective.replace("90 项", f"{target} 项").replace("达到 90", f"达到 {target}")
        objective = objective.replace("当前种子 101 的实验目标", "本次运行显式配置的实验目标")
    instructions = INSTRUCTIONS + "\n" + guide + EVENT_CONTEXT + "\n" + read(MULTIOBJECTIVE / "ONLINE_FIELDS.md")
    # Preserve the separator in the audited 100-second prompt.
    tools = read(HERE / "TOOLS.md").replace("\n\n", "\n \n", 1)
    return instructions + "\n" + tools, objective


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("codex", "smoke", "mock"), default="codex",
                   help="Smoke uses real Codex and exercises both queries on the first step.")
    p.add_argument("--steps", type=int, help="Physical steps, up to the native 7200-second horizon.")
    p.add_argument("--output", required=True, type=Path, help="New experiment output directory.")
    p.add_argument("--codex-home", type=Path, help="Already authenticated, isolated Codex home without MCP.")
    p.add_argument("--codex-bin", default="codex")
    p.add_argument("--model", default="gpt-6.1-sol")
    p.add_argument("--reasoning-effort", default="low")
    p.add_argument("--target-completed", type=int, help="Optional delivery floor (default 90 only at 100 steps).")
    p.add_argument("--step-timeout", type=float, default=90., help="Seconds across model turns and retries for one step.")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--wall-budget-seconds", type=float, help="Whole-run wall clock budget (default 12 seconds per step).")
    p.add_argument("--max-model-turns", type=int, help="Default (2 + retries) times the physical steps.")
    p.add_argument("--token-stop-threshold", type=int, help="Default 160000 reported tokens per step in aggregate.")
    p.add_argument("--thread-steps", type=int, default=100,
                   help="Start a fresh Codex thread after this many physical steps. Use 0 for one thread.")
    return p


def resolve(args, parser, scenario):
    if args.steps is None:
        args.steps = 2 if args.mode == "smoke" else 3600
    if args.mode != "mock" and (args.codex_home is None or not args.codex_home.is_dir()):
        parser.error("a real run requires an existing --codex-home")
    dt = scenario.environment.physical_step_seconds
    if not 1 <= args.steps <= scenario.observation_end_seconds // dt:
        parser.error("--steps is outside the native scenario horizon")
    if args.retries < 0 or args.thread_steps < 0:
        parser.error("--retries and --thread-steps must be nonnegative")
    target = args.target_completed
    if target is not None and target < 1:
        parser.error("--target-completed must be positive")
    if target is None and args.steps * dt == ORIGINAL_SECONDS:
        target = ORIGINAL_TARGET
    wall = args.wall_budget_seconds if args.wall_budget_seconds is not None else 12. * args.steps
    turns = args.max_model_turns if args.max_model_turns is not None else (2 + args.retries) * args.steps
    tokens = args.token_stop_threshold if args.token_stop_threshold is not None else 160_000 * args.steps
    if not all(math.isfinite(v) and v > 0 for v in (wall, args.step_timeout)):
        parser.error("timeouts and wall budget must be positive and finite")
    if turns < 1 or tokens < 1:
        parser.error("model-turn and token budgets must be positive")
    return dict(scenario=str(SCENARIO.relative_to(ROOT)), physical_step_seconds=dt, physical_steps=args.steps,
                workload_seed=SEED, scheduling_seed=SEED, condition_sha256=CONDITION, model=args.model,
                reasoning_effort=args.reasoning_effort, timeout_seconds=args.step_timeout, retries=args.retries,
                history_steps=1, wall_budget_seconds=wall, max_model_attempts=turns,
                reported_total_tokens_stop_threshold=tokens, thread_steps=args.thread_steps,
                completion_target=target, evaluation_cohort=dict(job_ids=list(COHORT), deadline_seconds=100),
                mode=args.mode)


def run(args, p, scenario):
    if scenario.metadata["condition_sha256"] != CONDITION or scenario.metadata["workload_seed"] != SEED:
        raise ValueError("Scenario identity differs from the audited Harness input")
    config, weights = _native_inputs(scenario, None, None)
    reward = load_reward_config(REWARD_CONFIG)
    env = FlexDCEnergyEnv(scenario, it_policy_config=config, weights=weights,
                          scheduling_seed=SEED, output_root=args.output)
    initial = observe(env, scenario)
    batch = [r for r in initial["unfinished_tasks"]["rows"] if r[1] == 0 and r[3] == 0]
    if tuple(r[0] for r in batch) != COHORT or any(r[3] + r[5] != 100 for r in batch):
        raise ValueError("Initial fixed batch differs from the audited Harness input")
    instructions, objective = guidance(p["physical_steps"] * p["physical_step_seconds"], p["completion_target"])
    output = Path(env.output_dir)
    write(output / "experiment_protocol.json", p)
    (output / "objective.md").write_text(objective + "\n", encoding="utf-8")
    (output / "model_instructions.md").write_text(instructions + "\n", encoding="utf-8")
    write(output / "instructions_sha256.json", {key: sha256(value.encode()).hexdigest()
          for key, value in (("objective", objective), ("model_instructions", instructions))})
    policy = MockPolicy() if args.mode == "mock" else HarnessPolicy(
        p, OnlinePayload(objective), model=args.model, effort=args.reasoning_effort,
        schema=decision_schema(env.get_action_spec()), executable=args.codex_bin, codex_home=args.codex_home,
        instructions=instructions, require_same_thread=True, force_smoke_query=args.mode == "smoke")
    try:
        summary = run_episode(env, scenario, policy, steps=p["physical_steps"], timeout_seconds=p["timeout_seconds"],
                              retries=p["retries"], history_steps=p["history_steps"], reward_config=reward)
    finally:
        if isinstance(policy, HarnessPolicy):
            policy.close()
    print(dumps({k: summary[k] for k in ("status", "steps_completed", "simulation_seconds", "model", "token_usage", "error")}))
    print(f"Logs: {output}")
    return 0 if summary["steps_completed"] == p["physical_steps"] and summary["status"] in (
        "step_limit", "scenario_terminal") else 1


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        scenario = load_scenario(SCENARIO)
        return run(args, resolve(args, cli, scenario), scenario)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FlexDC Harness failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
