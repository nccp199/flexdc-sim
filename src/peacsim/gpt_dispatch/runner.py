"""Advance FlexDC only after a successful decision at the current boundary."""
from collections import Counter, deque
import math
from pathlib import Path
from time import monotonic

from .decisions import INSTRUCTIONS, decision_schema, dumps, validate_decision
from .observation import observe, policy_contract
from peacsim.rl.rewards import TeamRewardLedger


def _decide(env, policy, payload, attempts, timeout_seconds, retries):
    deadline = monotonic() + timeout_seconds
    for index in range(retries + 1):
        started = monotonic()
        remaining = deadline - started
        if remaining <= 0:
            break
        budget = remaining / (retries + 1 - index)
        receipt = policy.decide(payload, timeout_seconds=budget)
        attempt = dict(attempt=index + 1, validation_feedback=payload["validation_feedback"],
                       inference_seconds=monotonic() - started, **receipt)
        attempts.append(attempt)
        if "error" not in receipt:
            try:
                decision, decoded = validate_decision(receipt["raw_output"], env)
            except (ValueError, TypeError) as exc:
                attempt["error"] = f"Invalid action: {exc}"
            else:
                attempt["accepted"] = True
                return decision, decoded
        attempt["accepted"] = False
        payload = {**payload, "validation_feedback": attempt["error"]}
        if not receipt.get("retryable", True):
            break
    reason = attempts[-1]["error"] if attempts else "no inference time remained"
    raise RuntimeError(f"No valid decision at step {payload['step']}. {reason}")


def _feedback(env, rows, next_state, reward, cumulative_limits):
    control = env.get_last_control()
    transition = env.get_last_transition()
    limits = Counter(reason for row in rows for reason in row["execution_limits"])
    cumulative_limits.update(limits)
    # Keep both physical requests and realized values. Native AQA can queue
    # commands at this boundary while the old commands are applied here.
    battery = [
        dict(start_seconds=row["start_seconds"], end_seconds=row["end_seconds"],
             requested=row["request"], after_no_export=row["facility_request"],
             executed=dict(charge_power_kw=row["power_kw"]["battery_charge"],
                           discharge_power_kw=row["power_kw"]["battery_discharge"]),
             reasons=row["execution_limits"])
        for row in rows
    ]
    result = dict(
        start_seconds=rows[0]["start_seconds"],
        end_seconds=rows[-1]["end_seconds"],
        submitted_action=control["policy_action"]["original_policy_action"],
        native_request=control["action"],
        task_weights=control.get("task_weights"),
        uniform_zero_preference_mapping=control["policy_action"]["zero_preference_fallback"],
        battery_execution=battery,
        power_intervals=[
            dict(start_seconds=row["start_seconds"], end_seconds=row["end_seconds"],
                 actual_it_kw=row["power_kw"]["it"], actual_grid_import_kw=row["power_kw"]["grid_import"],
                 pv_used_kw=row["power_kw"]["pv_used"], grid_target_kw=row["signals"]["grid_target_kw"],
                 grid_error_kw=row["grid_error_kw"], grid_purchase_cost=row["grid_purchase_cost"],
                 grid_carbon_kg=row["grid_carbon_kg"]) for row in rows
        ],
        it_actuation=dict(
            policy_sample_seconds=transition["policy_input"]["time_seconds"],
            aqa_received_budget_kw=transition["policy_input"]["cluster_target_power_w"] / 1000,
            prior_node_commands_applied=[
                [int(node), float(power)]
                for node, applied, power in zip(transition["before"]["node_id"],
                                                 transition["commands"]["applied_mask"],
                                                 transition["commands"]["applied_power_w"])
                if applied
            ],
            actual_it_kw_at_end=next_state["servers"]["actual_it_kw"],
        ),
        task_events={key: transition["events"][key].tolist()
                     for key in ("started_job_ids", "completed_job_ids")},
        task_counts=next_state["task_counts"],
        battery_soc_after=next_state["battery"]["soc"],
        reward=reward["team_reward"],
        battery_limit_events=dict(limits),
    )
    if "operating_conditions" in rows[0]:
        for power, row in zip(result["power_intervals"], rows):
            power["price_per_kwh"] = row["signals"]["price_per_kwh"]
        result["operating_conditions_executed"] = [
            dict(t=row["start_seconds"], **row["operating_conditions"],
                 grid_protection=row.get("grid_protection"),
                 priority_job_ids_at_admission=row["priority_job_ids_at_admission"])
            for row in rows
        ]
    return result


def _write_json(path, value):
    path.write_text(dumps(value, indent=2) + "\n", encoding="utf-8")


def _report(path, summary):
    metrics = summary["metrics"]
    cost = metrics["grid_purchase_cost"]
    path.write_text(
        "# FlexDC dispatch experiment\n\n"
        f"- Mode and model — {summary['method']}, {summary['model']}\n"
        f"- Status — {summary['status']}\n"
        f"- Physical steps completed — {summary['steps_completed']} of {summary['steps_requested']}\n"
        f"- Simulated time — {summary['simulation_seconds']} s\n"
        f"- Measured decision latency — {summary['decision_wall_seconds']:.3f} s\n"
        f"- Grid purchase cost — {cost['amount']:.9g} {cost['currency']}\n"
        f"- Tasks completed — {metrics['task_counts']['completed']}\n"
        f"- Task deadline violations to date — {metrics['task_counts']['deadline_violations_to_date']}\n"
        f"- Native battery limits applied — {metrics['battery_limit_events']}\n"
        f"- Scenario terminal evaluation settled — {summary['scenario_terminal']}\n"
        + (f"\nFailure — {summary['error']}\n" if summary["error"] else ""),
        encoding="utf-8",
    )


def run_episode(env, scenario, policy, *, steps, timeout_seconds, retries, history_steps, reward_config):
    """Own a fresh episode's journal. Exhausted validation/transport retries halt it."""
    if any(type(value) is not int or value < 1 for value in (steps, history_steps)):
        raise ValueError("steps and history_steps must be positive integers")
    if type(retries) is not int or retries < 0 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("retries must be nonnegative and the inference budget must be positive and finite")
    output = Path(env.output_dir)
    initial_energy = env.energy_summary()
    if initial_energy["covered_end_seconds"] != 0 or (output / "dispatch_config.json").exists():
        raise ValueError("dispatch requires a fresh episode directory")
    ledger = TeamRewardLedger(scenario, reward_config, run_id=initial_energy["run_id"])
    state = observe(env, scenario)
    contract = policy_contract(env, scenario, ledger.get_spec())
    history = deque(maxlen=history_steps)
    dt = contract["physical_step_seconds"]
    _write_json(output / "dispatch_config.json", dict(
        method=policy.method, model=policy.model, reasoning_effort=getattr(policy, "effort", None),
        max_output_tokens=getattr(policy, "max_output_tokens", None),
        scenario_id=scenario.scenario_id, condition_sha256=scenario.metadata["condition_sha256"],
        scenario_input_sha256=scenario.metadata["input_sha256"],
        workload_seed=scenario.metadata["workload_seed"],
        scheduling_seed=initial_energy["scheduling_seed"],
        native_initialization=initial_energy["native_controller"],
        steps=steps, timeout_seconds=timeout_seconds, retries=retries, history_steps=history_steps,
        model_instructions=getattr(policy, "instructions", INSTRUCTIONS),
        require_same_thread=getattr(policy, "require_same_thread", False),
        decision_schema=decision_schema(contract["action_spec"]),
        contract=contract, reward_spec=ledger.get_spec(),
    ))
    completed = 0
    inference_seconds = 0.0
    tokens = Counter()
    usage_records = 0
    cumulative_limits = Counter()
    error = None
    status = "running"
    terminal_report = None
    started_run = monotonic()

    def metrics():
        energy = env.energy_summary()
        return dict(grid_purchase_cost=energy["grid_purchase_cost"], grid_carbon_kg=energy["grid_carbon_kg"],
                    grid_tracking=energy.get("grid_tracking"), energy_kwh=energy["energy_kwh"],
                    task_counts=state["task_counts"], battery_soc=state["battery"]["soc"],
                    battery_limit_events=dict(cumulative_limits), reward=ledger.summary())

    try:
        with (output / "steps.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            for step in range(1, steps + 1):
                if state["terminal"]:
                    break
                payload = dict(step=step, experiment_steps_remaining=steps - step + 1,
                               contract=contract, observation=state,
                               recent_decisions=list(history), validation_feedback=None)
                record = dict(step=step, simulation_start_seconds=state["simulation_seconds"],
                              method=policy.method, model=policy.model, decision_input=payload,
                              attempts=[], submitted_action=None, safety=None, execution=None,
                              status="awaiting_decision")
                try:
                    decision, decoded = _decide(env, policy, payload, record["attempts"],
                                                timeout_seconds, retries)
                    action = decoded["original_policy_action"]
                    record.update(decision=decision, submitted_action=action,
                                  safety=dict(validation="native decoder accepted", decoded_action=decoded,
                                              adapter_changed_action=False),
                                  status="advancing_physics")
                    if not env.step_physical_actions(action["compute"], action["storage"]):
                        raise RuntimeError("Native physical step did not advance")
                    rows = env.get_last_energy_rows()
                    if rows[-1]["end_seconds"] - record["simulation_start_seconds"] != dt:
                        raise RuntimeError("Native step did not span exactly one physical interval")
                    completed += 1
                    record.update(native_energy_rows=rows, status="recording_feedback")
                    state = observe(env, scenario)
                    if state["terminal"]:
                        terminal_report = env.finalize()["episode"]
                    reward = ledger.append(rows, terminal_report=terminal_report)
                    execution = _feedback(env, rows, state, reward, cumulative_limits)
                    record["safety"]["native_battery_execution"] = execution["battery_execution"]
                    record.update(execution=execution, reward=reward,
                                  cumulative=metrics(), next_observation=state, status="executed")
                    history.append(dict(step=step, explanation=decision["explanation"],
                                        analysis=decision["analysis"], execution=execution,
                                        cumulative=dict(grid_purchase_cost=record["cumulative"]["grid_purchase_cost"],
                                                        battery_limit_events=dict(cumulative_limits),
                                                        reward_to_date=ledger.summary()["total_reward"])))
                except (Exception, KeyboardInterrupt) as exc:
                    record.update(status="failed", failed_phase=record["status"],
                                  error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    record["inference_seconds"] = sum(a["inference_seconds"] for a in record["attempts"])
                    inference_seconds += record["inference_seconds"]
                    for attempt in record["attempts"]:
                        if attempt.get("usage") is not None:
                            tokens.update({name: value for name, value in attempt["usage"].items()
                                           if type(value) is int})
                            usage_records += 1
                    stream.write(dumps(record) + "\n")
        status = "scenario_terminal" if state["terminal"] else "step_limit"
    except (Exception, KeyboardInterrupt) as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary = dict(
            status=status, error=error, method=policy.method, model=policy.model,
            steps_completed=completed, steps_requested=steps, physical_step_seconds=dt,
            simulation_seconds=env.energy_summary()["covered_end_seconds"],
            scenario_terminal=terminal_report is not None and ledger.summary()["complete"],
            decision_wall_seconds=inference_seconds, elapsed_wall_seconds=monotonic() - started_run,
            token_usage=dict(tokens) if usage_records else None, token_usage_records=usage_records,
            metrics=metrics(),
        )
        _write_json(output / "summary.json", summary)
        _report(output / "report.md", summary)
    return summary
