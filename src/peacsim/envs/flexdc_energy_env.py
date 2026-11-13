"""Native AQA and energy metering, with optional S4c joint control.

IT policy configuration is explicitly separate from the scenario's grid target.
Joint control holds a budget/request over the scenario's common control period.
The optional coordinator uses current signals only; RL observations are S5 work.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from peacsim.aqa_runtimepolicy import AQARuntimePolicy
from peacsim.control import JointAction, coordinate_power
from peacsim.energy.contracts import energy_interval_from_transition
from peacsim.energy.facility import BatteryRequest, FacilityEnergy
from peacsim.energy.import_limit import protect_import
from peacsim.energy.renewables import PV
from peacsim.metrics.tracking import _number
from peacsim.metrics.terminal import (
    EPISODE_REPORT_VERSION, TERMINAL_ENERGY_RULE, background_terminal_report, terminal_energy_condition,
)
from peacsim.scenarios import ScenarioBundle
from peacsim.simulation_session import SimulationSession
from peacsim.simulator import Simulator


class FlexDCEnergyEnv:
    """A fresh output_root owns episodes; reset preserves previous episode files.

    Construction/reset processes native t=0 once (no elapsed energy). Each step
    then advances one physical interval with a request provided before advance.
    step preserves the S4b native mode. step_joint/step_grid hold commands over
    a control period; choose one mode per episode (reset allows another mode).
    Native t=0 initialization is retained. A command issued at t reaches AQA
    at t+dt, with its existing further actuator delay; BESS serves [t,t+dt).
    """
    def __init__(self, scenario, *, it_policy_config, weights, scheduling_seed, output_root):
        if not isinstance(scenario, ScenarioBundle):
            raise TypeError('scenario must be a ScenarioBundle')
        if it_policy_config.runtime_policy_name != 'AQA':
            raise ValueError('the energy environment reuses the approved native AQA path')
        if (isinstance(scheduling_seed, bool) or not isinstance(scheduling_seed, int)
                or not 0 <= scheduling_seed < 2**32):
            raise ValueError('scheduling_seed must be an integer in [0, 2**32)')
        base = _number(it_policy_config.P, 'IT base power W', True)
        reserve = _number(it_policy_config.R, 'IT reserve power W', True)
        if not np.isfinite([base + reserve*v for v in scenario.iso_samples]).all():
            raise ValueError('IT targets must be finite')
        if weights is None:
            raise ValueError('supply resolved native AQA weights separately from the scenario')
        self._scenario = scenario
        self._condition_sha256 = scenario.metadata['condition_sha256']
        self._controller = dict(power_boundary='IT', base_w=base, reserve_w=reserve,
                                policy_parameters=deepcopy(it_policy_config.policy_parameters),
                                weights=deepcopy(weights))
        self._scheduling_seed = scheduling_seed
        self._default_scheduling_seed = scheduling_seed
        exp, jobs = scenario.experiment_config, scenario.job_config
        table, nodes = scenario.initial_jobs, scenario.initial_nodes
        policy = AQARuntimePolicy(deepcopy(it_policy_config), exp, jobs, nodes, table, weights=weights)
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=False)
        # Unstarted template creates no trajectory; Session owns the episode
        # directories, array copies, output initialization and reset behavior.
        prepared = Simulator(table, nodes, policy, exp, jobs, SimpleNamespace(),
                             output_dir=str(root)+'/', resolved_iso_signal=scenario.iso_samples)
        prepared._random_seed = scheduling_seed  # Existing documented native override.
        self._session = SimulationSession(prepared, root)
        self._begin_episode()

    def _begin_episode(self):
        initial = self._session.get_state()
        self._run_id = initial['meta']['episode_id']
        self._facility = FacilityEnergy(self._scenario, self._run_id)
        self._time = 0
        self._failed = self._finalized = False
        self._episode_report = None
        self._last_rows = ()
        self._mode = None
        self._last_control = None
        self._grid_abs_error_seconds = self._grid_max_error = 0.
        self._grid_duration = 0
        self._trace = Path(self._session.output_dir) / 'energy_trace.jsonl'
        self._trace.touch(exist_ok=False)
        try:
            self._session.advance_cycle()  # Reuse native initialization semantics.
        except Exception:
            self._failed = True
            raise

    @property
    def output_dir(self):
        return self._session.output_dir

    def reset(self, *, seed=None):
        self._session.reset(seed=seed)
        self._scheduling_seed = self._default_scheduling_seed if seed is None else seed
        self._begin_episode()
        return self.get_state()

    def step(self, request=BatteryRequest()):
        """One physical step with the original ISO-derived IT target."""
        if self._mode == 'joint':
            raise RuntimeError('use joint control throughout this episode, or reset')
        return self._advance_physical(request)

    def _advance_physical(self, request, *, action=None, control=None, task_weights=None):
        if self._failed:
            raise RuntimeError('failed run; reset before continuing')
        if not isinstance(request, BatteryRequest):
            raise TypeError('request must be a BatteryRequest selected before stepping')
        if self._time == self._scenario.observation_end_seconds:
            return False
        try:
            self._mode = 'joint' if action is not None else 'native'
            budget_w = None if action is None else action.it_power_budget_kw * 1000.
            conditions = self._scenario.operating_conditions(self._time)
            protected, protection, options = request, None, {}
            if conditions is not None:
                options['priority_types'] = conditions['priority_types']
                if conditions['grid_limit_kw'] is not None:
                    signal = self._scenario.at(self._time)
                    cfg = self._scenario.environment
                    protected, target, protection = protect_import(
                        limit=conditions['grid_limit_kw'],
                        it_kw=self._session.get_state()['derived']['actual_it_power_w'] / 1000,
                        cooling_ratio=cfg.cooling.power_ratio,
                        pv_kw=PV(cfg.pv).available_power_kw(signal['pv_fraction']), request=request,
                        battery_limits=self._facility.battery_power_limits(cfg.physical_step_seconds))
                    if target < protection['unprotected_it_kw'] - 1e-9:
                        options['immediate_it_limit_w'] = target * 1000
            if not self._session.advance_cycle(it_power_budget_w=budget_w, task_weights=task_weights, **options):
                raise RuntimeError('native run ended before the declared energy window')
            transition = self._session.get_last_transition()
            interval = energy_interval_from_transition(transition, run_id=self._run_id,
                                                      expected_episode_id=self._run_id)
            rows = self._facility.advance(interval, protected)
            if conditions is not None:
                priority_jobs = self._session.priority_jobs()
                for row in rows:
                    row.update(operating_conditions=conditions, priority_job_ids_at_admission=priority_jobs)
                    if protection is not None:
                        actual = row['power_kw']
                        violation = max(0., actual['grid_import'] - conditions['grid_limit_kw'])
                        row['grid_protection'] = dict(
                            **protection, actual_it_kw=actual['it'], actual_grid_kw=actual['grid_import'],
                            violation_kw=violation if violation > 1e-9 else 0.,
                            infeasible_reason='minimum_node_power_and_available_battery'
                            if violation > 1e-9 else None)
            if control is not None:
                for row in rows:
                    error = row['power_kw']['grid_import'] - row['signals']['grid_target_kw']
                    row.update(control=deepcopy(control), grid_error_kw=error)
                    seconds = row['end_seconds'] - row['start_seconds']
                    self._grid_abs_error_seconds += abs(error) * seconds
                    self._grid_max_error = max(self._grid_max_error, abs(error))
                    self._grid_duration += seconds
            with self._trace.open('a', encoding='utf-8') as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            self._time = interval.end_seconds
            self._last_rows = rows
        except Exception:
            self._failed = True
            raise
        return True

    def plan_grid_action(self, request=BatteryRequest()):
        """Read-only current-PV hold estimate; the caller chooses BESS direction."""
        if self._failed or self._time >= self._scenario.observation_end_seconds:
            raise RuntimeError('planning requires a valid unfinished episode')
        duration = min(self._scenario.environment.control_interval_seconds,
                       self._scenario.observation_end_seconds - self._time)
        signal = self._scenario.at(self._time)
        cfg = self._scenario.environment
        exp, jobs = self._scenario.experiment_config, self._scenario.job_config
        return coordinate_power(
            grid_target_kw=signal['grid_target_kw'],
            pv_available_kw=PV(cfg.pv).available_power_kw(signal['pv_fraction']),
            cooling_power_ratio=cfg.cooling.power_ratio,
            it_min_power_kw=exp.idle_power * exp.server_count / 1000.,
            it_max_power_kw=max(exp.idle_power, max(jobs.all_max_job_power.values())) * exp.server_count / 1000.,
            battery_request=request, battery_limits=self._facility.battery_power_limits(duration))

    def step_grid(self, request=BatteryRequest()):
        """Optional coordinator; independent agents may call step_joint instead."""
        if self._time == self._scenario.observation_end_seconds and not self._failed:
            return False
        plan = self.plan_grid_action(request)
        return self._advance_control(plan.action, plan=plan)

    def step_joint(self, action, *, task_weights=None):
        """Hold one externally selected joint command over a control period."""
        return self._advance_control(action, task_weights=task_weights)

    def get_action_spec(self):
        """S5b static policy-action dimensions, bounds, mappings and probability contract."""
        from peacsim.rl.actions import action_spec
        exp = self._scenario.experiment_config
        return action_spec(job_config=self._scenario.job_config,server_count=exp.server_count,
                           idle_power_w=exp.idle_power,environment=self._scenario.environment)

    def decode_actions(self, compute, storage):
        """Read-only conversion; physical constraints are applied during execution."""
        from peacsim.rl.actions import decode_actions
        return decode_actions(compute,storage,spec=self.get_action_spec())

    def step_actions(self, compute, storage):
        """Execute a policy-action pair for the existing common control period."""
        return self._step_policy_actions(compute, storage)

    def step_physical_actions(self, compute, storage):
        """Execute the same policy-action pair for one native physical step."""
        return self._step_policy_actions(compute, storage, physical_step=True)

    def _step_policy_actions(self, compute, storage, *, physical_step=False):
        if self._failed:
            raise RuntimeError('failed run; reset before continuing')
        if self._time == self._scenario.observation_end_seconds:
            return False
        decoded = self.decode_actions(compute,storage)
        return self._advance_control(decoded.joint_action,task_weights=decoded.task_weights,
                                     policy_action=decoded.to_record(), physical_step=physical_step)

    def _advance_control(self, action, *, plan=None, task_weights=None, policy_action=None,
                         physical_step=False):
        if self._failed:
            raise RuntimeError('failed run; reset before continuing')
        if not isinstance(action, JointAction):
            raise TypeError('action must be a JointAction')
        if self._mode == 'native':
            raise RuntimeError('use native control throughout this episode, or reset')
        if self._time == self._scenario.observation_end_seconds:
            return False
        if task_weights is not None:
            from peacsim.control.allocation import normalize_task_weights
            task_weights = normalize_task_weights(task_weights,self._scenario.job_config.job_type_count)
        cfg = self._scenario.environment
        period = cfg.physical_step_seconds if physical_step else cfg.control_interval_seconds
        end = min(self._time + period, self._scenario.observation_end_seconds)
        control = dict(issued_at_seconds=self._time, hold_until_seconds=end,
                       first_aqa_sample_seconds=self._time + cfg.physical_step_seconds,
                       last_aqa_sample_seconds=end, action=asdict(action))
        if task_weights is not None:
            control['task_weights'] = list(task_weights)
        record = dict(**control, prediction=None if plan is None else asdict(plan),
                      prediction_method=None if plan is None else 'current_pv_hold_no_future_lookup')
        if policy_action is not None:
            record['policy_action'] = policy_action
        rows = []
        try:
            with (Path(self.output_dir) / 'control_trace.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            while self._time < end:
                self._advance_physical(action.battery_request, action=action, control=control,
                                       task_weights=task_weights)
                rows.extend(self._last_rows)
        except Exception:
            self._failed = True
            raise
        self._last_rows = tuple(rows)
        self._last_control = record
        return True

    def get_last_control(self):
        return deepcopy(self._last_control)

    def get_state(self):
        return dict(native=self._session.get_state(), energy=self.energy_summary())

    def get_observation_spec(self):
        """S5a legal field dictionary; independent of task population and RNG."""
        from peacsim.rl.specs import observation_spec
        exp = self._scenario.experiment_config
        return observation_spec(
            job_config=self._scenario.job_config, server_count=exp.server_count,
            idle_power_w=exp.idle_power, environment=self._scenario.environment,
            observation_end_seconds=self._scenario.observation_end_seconds,
            price_unit=self._scenario.metadata['signals']['price_per_kwh']['unit'])

    def get_observations(self, *, physical_step=False):
        """S5a read-only Actor observations and their legal union for the Critic."""
        from peacsim.rl.observations import build_observations
        return build_observations(**self._observation_inputs(physical_step=physical_step))

    def _observation_inputs(self, *, physical_step=False):
        """Capture one current boundary, shared by legacy and S7b readers."""
        if self._failed:
            raise RuntimeError('failed run; reset before observing')
        native = self._session.get_state()
        if native['meta']['time_seconds'] != self._time:
            raise RuntimeError('native and energy observation clocks must agree')
        remaining = self._scenario.observation_end_seconds - self._time
        config = self._scenario.environment
        if physical_step:
            # Only the decision horizon changes, for BESS limits and task urgency.
            config = replace(config, control_interval_seconds=config.physical_step_seconds)
        duration = min(config.control_interval_seconds, remaining)
        return dict(
            native=native, job_config=self._scenario.job_config,
            environment=config,
            observation_end_seconds=self._scenario.observation_end_seconds,
            current_signals=self._scenario.at(self._time) if remaining else None,
            battery_current=self._facility.summary()['battery_current'],
            battery_limits=self._facility.battery_power_limits(duration) if duration else None,
            last_energy_rows=self._last_rows, spec=self.get_observation_spec())

    def get_critic_inputs(self):
        """S7b common summary (legacy fields plus six task-history counts)."""
        from peacsim.rl.observations import build_observations
        from peacsim.rl.critic_state import build_critic_summary
        inputs=self._observation_inputs()
        return build_critic_summary(native=inputs['native'],observed=build_observations(**inputs),
            observation_spec=inputs['spec'],cohort_seconds=self._scenario.cohort_seconds,
            endpoint=self._scenario.observation_end_seconds)

    def get_critic_graph(self):
        """S7b graph with exactly the same public summary, no learning side effects."""
        from peacsim.rl.observations import build_observations
        from peacsim.rl.graphs import build_graph
        inputs=self._observation_inputs()
        return build_graph(native=inputs['native'],observed=build_observations(**inputs),
            observation_spec=inputs['spec'],job_config=inputs['job_config'],
            environment=inputs['environment'],cohort_seconds=self._scenario.cohort_seconds,
            endpoint=self._scenario.observation_end_seconds,idle_power_w=self._scenario.experiment_config.idle_power)

    def get_last_energy_rows(self):
        return deepcopy(self._last_rows)

    def get_operating_conditions(self):
        return self._scenario.operating_conditions(self._time)

    def get_last_transition(self):
        return self._session.get_last_transition()

    def energy_summary(self):
        result = self._facility.summary()
        result.update(valid=result['valid'] and not self._failed,
                      scenario_id=self._scenario.scenario_id,
                      condition_sha256=self._condition_sha256,
                      native_controller=deepcopy(self._controller), scheduling_seed=self._scheduling_seed,
                      terminal_soc_rule=TERMINAL_ENERGY_RULE,
                      joint_grid_control=self._mode == 'joint')
        result['terminal_energy_condition'] = (terminal_energy_condition(
            result['battery']['initial_energy_kwh'], result['battery']['final_energy_kwh'])
            if result['complete'] and result['valid'] else None)
        if self._mode == 'joint':
            result.update(control_interval_seconds=self._scenario.environment.control_interval_seconds,
                          grid_tracking=dict(duration_seconds=self._grid_duration,
                                             mean_absolute_error_kw=(self._grid_abs_error_seconds / self._grid_duration
                                                                     if self._grid_duration else None),
                                             max_absolute_error_kw=self._grid_max_error))
        return result

    def finalize(self):
        if self._failed or not self._facility.summary()['complete']:
            raise RuntimeError('cannot finalize an invalid or incomplete energy run')
        if self._finalized:
            raise RuntimeError('already finalized')
        try:
            native_result = self._session.finalize()
            report = self.energy_summary()
            directory = Path(self.output_dir)
            tasks, background = background_terminal_report(
                self._session.get_job_report(), self._scenario.job_config,
                cohort_seconds=self._scenario.cohort_seconds, observed_until=self._time)
            cohort = json.loads((directory/'qos_evaluation.json').read_text(encoding='utf-8'))
            episode = dict(schema_version=EPISODE_REPORT_VERSION, run_id=self._run_id,
                           condition_sha256=self._condition_sha256, physical_run_complete=True,
                           observation_end_seconds=self._time, cohort_seconds=list(self._scenario.cohort_seconds),
                           terminal_energy=report['terminal_energy_condition'],
                           background=background, cohort_qos=cohort, energy=report,
                           it_tracking=self._session.get_tracking_report())
            tasks.to_csv(directory/'background_task_status.csv', index=False)
            (directory/'episode_report.json').write_text(
                json.dumps(episode, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            path = directory / 'energy_summary.json'
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        except Exception:
            self._failed = True
            raise
        self._finalized = True
        self._episode_report = deepcopy(episode)
        return dict(native_result=native_result, energy=report, episode=self.get_episode_report())

    def get_episode_report(self):
        """Owned terminal report, available only after successful finalization."""
        if not self._finalized:
            raise RuntimeError('episode report is unavailable until finalize succeeds')
        return deepcopy(self._episode_report)
