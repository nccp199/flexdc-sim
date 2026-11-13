"""Run S4d rules through the public S4a/S4b/S4c interfaces, sequentially.

Native global RNG semantics require sequential episodes. Both rules use the
same coordinator, initialization, physical devices, signals and evaluation.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from peacsim.control.rule_controllers import RULE_NAMES, RULE_VERSION, decide_battery
from peacsim.energy.cooling import Cooling
from peacsim.energy.renewables import PV
from peacsim.envs import FlexDCEnergyEnv


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')


def run_rule_episode(scenario, *, rule, it_policy_config, weights, scheduling_seed, output_root):
    """Run one fixed rule to the existing common observation endpoint.

    Current native boundary IT power is a hold estimate of future load; it is
    never substituted into the actual energy ledger. The rule gets three
    scalar inputs only, not tasks, future curves, prices or a target error
    already affected by its previous battery action.
    """
    if rule not in RULE_NAMES:
        raise ValueError('unknown battery rule: ' + str(rule))
    env = FlexDCEnergyEnv(scenario, it_policy_config=it_policy_config, weights=weights,
                          scheduling_seed=scheduling_seed, output_root=output_root)
    directory = Path(env.output_dir)
    cooling, pv = Cooling(scenario.environment.cooling), PV(scenario.environment.pv)
    state = env.get_state()
    digest = hashlib.sha256()
    digest.update(state['native']['raw']['job_table'].tobytes())
    digest.update(state['native']['raw']['node_table'].tobytes())
    decisions = 0
    with (directory/'rule_trace.jsonl').open('x', encoding='utf-8') as stream:
        while not state['energy']['complete']:
            time = state['native']['meta']['time_seconds']
            signal = scenario.at(time)
            it = state['native']['derived']['actual_it_power_w'] / 1000.
            inputs = dict(estimated_facility_load_kw=it + cooling.power_kw(it),
                          pv_available_kw=pv.available_power_kw(signal['pv_fraction']),
                          grid_target_kw=signal['grid_target_kw'])
            decision = decide_battery(rule, **inputs)
            record = dict(rule_version=RULE_VERSION, issued_at_seconds=time,
                          load_estimate_source='current_native_boundary_power_hold',
                          current_it_power_kw=it, inputs=inputs,
                          battery_before=state['energy']['battery_current'], **asdict(decision))
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            # The single existing coordinator handles the BESS request before
            # allocating the IT budget. No second correction is added here.
            if not env.step_grid(decision.battery_request):
                raise RuntimeError('environment ended before its declared window')
            decisions += 1
            state = env.get_state()
    final = env.finalize()
    energy = final['energy']
    qos = json.loads((directory/'qos_evaluation.json').read_text(encoding='utf-8'))
    result = dict(schema_version='flexdc-rule-run-v2', rule_version=RULE_VERSION, rule=rule,
                  decision_count=decisions, initial_native_state_sha256=digest.hexdigest(),
                  output_dir=str(directory), energy=energy, qos=qos, episode=final['episode'],
                  terminal_settlement='initial-energy floor checked; no cost adjustment; background classified')
    _write_json(directory/'rule_summary.json', result)
    return result


def comparison_row(report):
    """Select existing metrics without recomputing QoS or physical accounts."""
    energy, qos = report['energy'], report['qos']
    battery = energy['battery']
    terminal, background = report['episode']['terminal_energy'], report['episode']['background']
    return dict(rule=report['rule'], output_dir=report['output_dir'],
                grid_mae_kw=energy['grid_tracking']['mean_absolute_error_kw'],
                grid_max_abs_error_kw=energy['grid_tracking']['max_absolute_error_kw'],
                grid_energy_kwh=energy['energy_kwh']['grid_import'],
                grid_purchase_cost=energy['grid_purchase_cost'], grid_carbon_kg=energy['grid_carbon_kg'],
                qos_total=qos['total'], qos_on_time=qos['on_time'], qos_late=qos['late'],
                qos_pending=qos['pending'], qos_final=qos['final'],
                battery_initial_energy_kwh=battery['initial_energy_kwh'],
                battery_final_energy_kwh=battery['final_energy_kwh'], battery_final_soc=battery['final_soc'],
                battery_net_energy_change_kwh=battery['final_energy_kwh']-battery['initial_energy_kwh'],
                battery_charge_kwh=energy['energy_kwh']['battery_charge'],
                battery_discharge_kwh=energy['energy_kwh']['battery_discharge'],
                pv_curtailed_kwh=energy['energy_kwh']['pv_curtailed'],
                terminal_energy_condition_met=terminal['condition_met'],
                terminal_energy_shortfall_kwh=terminal['shortfall_kwh'],
                background_total=background['total'],
                background_completed_on_time=background['completed_on_time'],
                background_completed_late=background['completed_late'],
                background_overdue_unfinished=background['overdue_unfinished'],
                background_not_due_unfinished=background['not_due_unfinished'],
                background_deadline_violations=background['deadline_violation_count'],
                background_backlog_waiting=background['backlog']['all']['waiting_count'],
                background_backlog_running=background['backlog']['all']['running_count'],
                background_backlog_queue_wait_max_seconds=background['backlog']['all']['queue_wait_max_seconds'])


def run_rule_comparison(scenario, *, it_policy_config, weights, scheduling_seed, output_root):
    """Run the two approved rules under one frozen set of experiment inputs."""
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    reports = [run_rule_episode(scenario, rule=rule, it_policy_config=it_policy_config,
                                weights=weights, scheduling_seed=scheduling_seed, output_root=root/rule)
               for rule in RULE_NAMES]
    conditions = []
    for report in reports:
        energy, qos = report['energy'], report['qos']
        conditions.append(dict(condition_sha256=energy['condition_sha256'],
                               scheduling_seed=energy['scheduling_seed'],
                               native_controller=energy['native_controller'],
                               initial_native_state_sha256=report['initial_native_state_sha256'],
                               cohort_sha256=qos['cohort_sha256'],
                               start_seconds=energy['start_seconds'], end_seconds=energy['end_seconds'],
                               initial_battery_energy_kwh=energy['battery']['initial_energy_kwh'],
                               terminal_rule_version=report['episode']['schema_version'],
                               control_interval_seconds=energy['control_interval_seconds']))
    if conditions[0] != conditions[1]:
        raise RuntimeError('rule comparison conditions differ')
    result = dict(schema_version='flexdc-rule-comparison-v2', rule_version=RULE_VERSION,
                  scenario_id=scenario.scenario_id, shared_conditions=conditions[0],
                  rows=[comparison_row(r) for r in reports],
                  interpretation='check terminal-energy condition, task service and tracking together; not-due backlog is diagnostic',
                  terminal_settlement='initial-energy floor checked; surplus is reported without cost credit')
    _write_json(root/'comparison.json', result)
    return result
