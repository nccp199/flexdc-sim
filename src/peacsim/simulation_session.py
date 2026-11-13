"""Stable episode interface; each reset constructs a fresh Simulator and AQA.

Freeze an unstarted, fully prepared Simulator (including any scheduling-seed
override). This layer never regenerates workloads, reselects ISO offsets, or
reads weight files. It does not own learning networks/optimizers. The existing
global RNG still requires sequential episodes or separate worker processes.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import shutil

from peacsim.aqa_runtimepolicy import AQARuntimePolicy
from peacsim.simulator import Simulator


class SimulationSession:
    def __init__(self, prepared_simulator, output_root):
        sim = prepared_simulator
        sim._require_cycle_boundary()
        if type(sim) is not Simulator or sim._next_cycle != 0 or sim._run_finalized:
            raise ValueError('Prepare an unstarted Simulator before creating a session')
        if type(sim._runtime_policy) is not AQARuntimePolicy:
            raise TypeError('Reconstruction currently supports only the approved AQARuntimePolicy')
        policy = sim._runtime_policy
        if policy._job_table is not sim._job_table or policy._node_table is not sim._node_table:
            raise ValueError('Simulator and policy must share their live task/node arrays')
        self._initial_jobs = sim._job_table.copy()
        self._initial_nodes = sim._node_table.copy()
        self._experiment = deepcopy(sim._experiment_config)
        self._jobs = deepcopy(sim._job_config)
        self._cluster = deepcopy(sim._cluster_config)
        self._signal = deepcopy(sim._iso_signal)
        self._weights = deepcopy(policy.weights)
        self._policy_config = SimpleNamespace(
            runtime_policy_name=policy._runtime_policy_name, P=policy.P, R=policy.R,
            policy_parameters=deepcopy(policy._policy_parameters))
        self._availability_qos = policy.availability_qos
        self._seed = sim._random_seed
        self._qos_percentile = sim.qos_violation_percentile_cutoff
        self._it_tariff = sim._it_tariff
        self._output_root = Path(output_root).resolve()
        self._episode_index = -1
        self._simulator = None
        self.reset()

    @property
    def output_dir(self):
        """Current episode directory (string with trailing slash for old writers)."""
        return self._simulator._output_dir

    def reset(self, *, seed=None):
        """Rebuild before t=0; return a snapshot of the new initial state.

        Omitted seed restores the frozen scheduling seed; an explicit seed only
        changes this episode's scheduler RNG, never its workload or signal.
        Policy parameters also remain frozen. A different candidate needs a new
        prepared experiment; reset is not an action/configuration update API.
        Previous episode files remain intact. Existing directories are refused.
        Publish the new episode only after construction and output setup succeed.
        """
        if self._simulator is not None and self._simulator._cycle_in_progress:
            raise RuntimeError('Reset is allowed only between cycles')
        chosen_seed = self._seed if seed is None else seed
        if isinstance(chosen_seed, bool) or not isinstance(chosen_seed, int) or not 0 <= chosen_seed < 2**32:
            raise ValueError('Scheduling seed must be an integer in [0, 2**32)')
        index = self._episode_index + 1
        path = self._output_root / ('episode_%06d' % index)
        path.mkdir(parents=True, exist_ok=False)
        try:
            exp, jobs, cluster = deepcopy((self._experiment, self._jobs, self._cluster))
            config = deepcopy(self._policy_config)
            job_table, node_table = self._initial_jobs.copy(), self._initial_nodes.copy()
            policy = AQARuntimePolicy(config, exp, jobs, node_table, job_table, weights=self._weights)
            policy.availability_qos = self._availability_qos
            fresh = Simulator(job_table, node_table, policy, exp, jobs, cluster,
                              qos_violation_percentile_cut_off=self._qos_percentile,
                              output_dir=str(path) + '/', resolved_iso_signal=self._signal,
                              it_tariff=self._it_tariff)
            fresh._random_seed = chosen_seed
            fresh._episode_id = str(path)  # Unique within/across session output roots; no RNG draw.
            from peacsim.simulation_output import init_output
            init_output(fresh._output_dir, exp, config)
            initial = fresh.get_state()
        except BaseException:
            # Only the directory exclusively created above belongs to this attempt.
            shutil.rmtree(path)
            raise
        self._simulator = fresh
        self._episode_index = index
        return initial

    def advance_cycle(self, *, it_power_budget_w=None, task_weights=None, **operating_constraints):
        # Manage the sink internally, so reset never has a borrowed file to close.
        return self._simulator.advance_cycle(it_power_budget_w=it_power_budget_w, task_weights=task_weights,
                                             **operating_constraints)

    def priority_jobs(self):
        return sorted(map(int, self._simulator._runtime_policy.priority_jobs))

    def run(self):
        return self._simulator.run()

    def finalize(self):
        return self._simulator.finalize()

    def get_state(self):
        return self._simulator.get_state()

    def get_last_transition(self):
        return self._simulator.get_last_transition()

    @staticmethod
    def get_transition_schema():
        return Simulator.get_transition_schema()

    def get_job_array(self):
        return self._simulator.get_job_array()

    def get_job_report(self):
        return self._simulator.get_job_report()

    def get_tracking_report(self):
        return self._simulator.get_tracking_report()

    def get_energy_cost_report(self):
        return self._simulator.get_energy_cost_report()

    @staticmethod
    def get_state_schema():
        return Simulator.get_state_schema()
