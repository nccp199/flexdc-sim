"""Project the current FlexDC boundary into model-visible JSON."""
from dataclasses import asdict

import numpy as np
import pandas as pd

from peacsim.qos_evaluation import evaluate_qos
from peacsim.rl.observations import task_details


OBSERVATION_VERSION = "flexdc-gpt-observation-v1"


def policy_contract(env, scenario, reward_spec):
    spec = env.get_action_spec()
    actions = {key: spec[key] for key in (
        "schema_version", "agent_order", "task_type_order", "dimensions", "fields",
        "active_dimensions", "inactive_canonical_value", "physical_bounds",
        "zero_preference_rule", "queue_rule",
    )}
    jobs = scenario.job_config
    return dict(
        action_spec=actions,
        physical_step_seconds=scenario.environment.physical_step_seconds,
        scenario_end_seconds=scenario.observation_end_seconds,
        price_unit=env.get_observation_spec()["fields"]["common.price_per_kwh"]["unit"],
        objective={key: reward_spec[key] for key in (
            "purpose", "objective", "components", "cohort_seconds", "task_aggregation", "initial_energy_kwh",
        )},
        task_types=[
            dict(type_id=j, name=jobs.all_jobs[j], required_nodes=jobs.all_job_size[j],
                 min_run_seconds=jobs.all_min_execution_time[j],
                 max_run_seconds=jobs.all_max_execution_time[j],
                 min_node_power_w=jobs.all_min_job_power[j],
                 max_node_power_w=jobs.all_max_job_power[j],
                 qos_allowance_ratio=jobs.all_job_qos_constraints[j])
            for j in actions["task_type_order"]
        ],
        battery=asdict(scenario.environment.battery),
        cooling_power_ratio=scenario.environment.cooling.power_ratio,
        grid_policy="import only, PV first; grid target is a tracking objective",
        it_budget_policy="AQA power target; task placement, minimum power and delayed caps are native",
        deadline_rule="completion at deadline is on time; unfinished after processing the deadline is late",
        signal_scope="current observations and completed intervals; forecasts are not supplied",
    )


def observe(env, scenario):
    # The native snapshot also has a raw pre-generated table. Only the named
    # arrived-task and current-node views below can cross the model boundary.
    state = env.get_state()
    native, energy = state["native"], state["energy"]
    observed = env.get_observations(physical_step=True)
    time_seconds = observed["time_seconds"]
    dt = scenario.environment.physical_step_seconds
    end = scenario.observation_end_seconds
    jobs = native["state"]["jobs"]
    nodes = native["state"]["nodes"]
    control = native["control"]
    profiles = scenario.job_config
    tasks = []
    for row in task_details(native, profiles, end, dt):
        i = row["index"]
        start = jobs["start_time_seconds"][i]
        minimum = float(jobs["min_execution_time_seconds"][i])
        tasks.append([
            row["job_id"], row["job_type"], "waiting" if row["waiting"] else "running",
            float(jobs["arrival_time_seconds"][i]), None if row["waiting"] else float(start),
            row["slack"], row["work"], 1.0 - row["work"] / (row["required_nodes"] * minimum),
            row["queue_rank"] if row["waiting"] else None,
        ])

    arrived = (jobs["arrival_time_seconds"] <= time_seconds) & (jobs["arrival_time_seconds"] < end)
    finishes = jobs["end_time_seconds"][arrived]
    population = pd.DataFrame(dict(
        arrival_time=jobs["arrival_time_seconds"][arrived],
        end_time=np.where(np.isnan(finishes), -1.0, finishes),
        job_type_id=jobs["job_type_id"][arrived],
    ))
    _, qos = evaluate_qos(
        population, profiles.job_type_count, profiles.all_job_qos_constraints,
        profiles.all_min_execution_time, time_seconds, 0, end,
    )
    counts = native["derived"]
    raw = observed["raw"]
    signals = None
    if raw["common.signals_valid"]:
        signals = {name: raw["common." + name] for name in (
            "grid_target_kw", "pv_available_kw", "price_per_kwh", "carbon_kg_per_kwh",
        )}
    limits = None if signals is None else dict(
        charge_kw=raw["common.charge_limit_kw"], discharge_kw=raw["common.discharge_limit_kw"],
    )
    last_rows = env.get_last_energy_rows()
    last_power = last_rows[-1]["power_kw"] if last_rows else {}
    result = dict(
        schema_version=OBSERVATION_VERSION,
        simulation_seconds=time_seconds,
        terminal=time_seconds == end,
        remaining_scenario_seconds=end - time_seconds,
        signals_observed_now=signals,
        forecasts=None,
        battery=dict(**energy["battery_current"], device_limits_for_step_kw=limits,
                     last_charge_kw=last_power.get("battery_charge"),
                     last_discharge_kw=last_power.get("battery_discharge")),
        last_grid_import_kw=last_power.get("grid_import"),
        facility_load_now_kw=raw["common.facility_load_kw"],
        servers=dict(count=len(nodes["node_id"]), busy=counts["busy_nodes"],
                     idle=counts["idle_nodes"], utilization=counts["busy_nodes"] / len(nodes["node_id"]),
                     actual_it_kw=counts["actual_it_power_w"] / 1000,
                     last_it_target_kw=control["cluster_target_power_w"] / 1000),
        task_counts=dict(arrived=int(arrived.sum()), waiting=sum(t[2] == "waiting" for t in tasks),
                         running=sum(t[2] == "running" for t in tasks),
                         completed=int(np.count_nonzero(~np.isnan(finishes))),
                         deadline_violations_to_date=int(qos["late"].sum())),
        rl_raw_summary=raw,
        unfinished_tasks=dict(
            columns=["id", "type_id", "status", "arrival_seconds", "start_seconds", "deadline_slack_seconds",
                     "remaining_work_node_seconds", "progress_ratio", "waiting_rank_in_type"],
            rows=tasks,
        ),
        nodes=dict(
            columns=["id", "task_id", "actual_power_w", "pending_command_w", "task_progress_ratio"],
            rows=[
                [int(node_id), int(nodes["job_id"][i]) if nodes["is_busy"][i] else None,
                 float(nodes["actual_power_w"][i]),
                 float(control["requested_power_w"][i]) if control["command_present"][i] else None,
                 (0.0 if nodes["just_started"][i] else float(nodes["task_progress_ratio"][i]))
                 if nodes["is_busy"][i] else None]
                for i, node_id in enumerate(nodes["node_id"])
            ],
        ),
    )
    conditions = env.get_operating_conditions()
    if conditions is not None:
        types = conditions["priority_types"]
        result["operating_conditions"] = dict(
            **conditions, priority_job_ids=[task[0] for task in tasks if task[1] in types])
    return result
