import math
import numpy as np
import pandas as pd
import os
import random
from copy import deepcopy
from tqdm import tqdm

from peacsim.iso_signal import read_iso_signal, calculate_cluster_power_caps
from peacsim.extract_qos_metrics import write_qos_summary_df
from peacsim.extract_power_metrics import get_tracking_error_90th, get_write_tracking_error_90th
from peacsim.transition_record import TransitionRecord
from peacsim.metrics.reporting import TrackingRecorder
from peacsim.metrics.energy_cost import ElectricityTariff

"""
Authors: @JoshBard @FatihAcun @EthanL3
Created 02/11/2024
PeacLab
"""

(node_id, server_job_id, server_state, server_power, estimate_power, job_progress, min_exec_time, max_exec_time,
 min_job_power, max_job_power, just_started, control_power) = range(12)

(job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time, realtime_qos,
 min_execution_time, qos_constraint) = range(10)

JOB_COLUMNS = ['job_id', 'job_type_id', 'arrival_time', 'start_time',
               'end_time', 'estimate_finished', 'update_time', 'realtime_qos',
               'min_execution_time', 'qos_constraint']


def calculate_unit_progress(running_nodes):
    """
    Function to calculate how much progress a job has made in a given timestep
    """
    return (1 / running_nodes[max_exec_time] + (
                1 / running_nodes[min_exec_time] - 1 / running_nodes[max_exec_time]) *
            ((running_nodes[server_power] - running_nodes[min_job_power]) / (
                        running_nodes[max_job_power] - running_nodes[min_job_power])))


class Simulator:
    def __init__(self, job_table, node_table, runtime_policy,
                 experiment_config, job_config, cluster_config, qos_violation_percentile_cut_off=0.9, output_dir=None,
                 *, resolved_iso_signal=None, it_tariff=None):

        if it_tariff is not None and not isinstance(it_tariff, ElectricityTariff):
            raise TypeError('it_tariff must be an ElectricityTariff or None')
        self._it_tariff = it_tariff
        self._energy_cost_report = None

        self.qos_emergency = None
        self.standby_power = None
        self.tracking_num = None
        self.waiting_sum = None
        self.qos_job_sum = None
        self.real_power = None
        self.tracking_error = None
        self.server_state_at_update = None
        self._job_config = job_config
        self._experiment_config = deepcopy(experiment_config)
        self._job_table = job_table
        self._finished_job_table = pd.DataFrame()
        self._node_table = node_table
        self._server_count = experiment_config.server_count
        self._job_type_count = job_config.job_type_count
        self._runtime_policy = runtime_policy
        self._cluster_config = cluster_config
        self._idle_status = -1

        self._time_granularity = experiment_config.time_granularity
        self._random_seed = experiment_config.random_seed
        self._simulation_duration = experiment_config.simulation_duration
        self._qos_cohort_start = getattr(experiment_config, 'qos_cohort_start', 0)
        self._qos_cohort_end = getattr(experiment_config, 'qos_cohort_end', self._simulation_duration)
        self._qos_boundary_snapshot = None
        self.qos_summary = None
        self._utilization = experiment_config.utilization
        self._normalize_iso_signal = experiment_config.normalize_iso_signal
        self._iso_signal_granularity = experiment_config.iso_signal_granularity
        self._idle_power = experiment_config.idle_power

        self._iso_signal = (read_iso_signal(experiment_config,
                                           self._simulation_duration,
                                           self._iso_signal_granularity,
                                           self._normalize_iso_signal)
                            if resolved_iso_signal is None else deepcopy(resolved_iso_signal))
        if len(self._iso_signal) < self._simulation_duration // self._iso_signal_granularity + 1:
            raise ValueError('Resolved ISO signal is shorter than the observation window')

        self._cluster_power_cap_per_signal = calculate_cluster_power_caps(self._iso_signal,
                                                                          self._runtime_policy.P,
                                                                          self._runtime_policy.R,
                                                                          0,
                                                                          self._simulation_duration,
                                                                          self._iso_signal_granularity)
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.tracking_error_at_90 = None
        self.qos_violation_percentile_cutoff = qos_violation_percentile_cut_off
        self.ratio_of_job_type_qos_violation = None
        self.constraint_violations = None
        self.waiting_jobs_by_job_type = []
        self.probabilistic_results = []
        self._next_cycle = 0
        self._run_finalized = False
        self._job_report = None
        self._final_costs = None
        self._episode_id = None  # Assigned by SimulationSession for managed episodes.
        self._cycle_in_progress = False
        self._cycle_failed = False
        self._last_transition = None
        self._last_policy_target_w = None
        self._transition_capture = None
        self._tracking_report = None
        self._tracking_recorder = TrackingRecorder(
            self._cluster_power_cap_per_signal, self._runtime_policy.R, self._runtime_policy.P,
            self._iso_signal_granularity, self._simulation_duration, self._time_granularity)

    def _require_cycle_boundary(self):
        if self._cycle_in_progress or self._cycle_failed:
            raise RuntimeError('State is not at a successful cycle boundary; rebuild/reset after a failed cycle.')

    def get_state(self):
        """Return an owned, on-demand diagnostic snapshot; see get_state_schema().

        Call on the advancing thread between successful cycles. Raw tables can
        contain future inputs; this is neither an RL observation nor a checkpoint.
        """
        self._require_cycle_boundary()
        from peacsim.state_snapshot import snapshot
        return snapshot(self)

    @staticmethod
    def get_state_schema():
        """Return an independent field dictionary with units, phases and validity."""
        from peacsim.state_snapshot import get_state_schema
        return get_state_schema()

    def get_last_transition(self):
        """Return an owned record of the last cycle, or None before t=0/reset.

        Read between cycles on the advancing thread. Only the latest cycle is
        kept; consume each successful advance if a full history is required.
        See get_transition_schema() for phases, validity and units.
        """
        self._require_cycle_boundary()
        return None if self._last_transition is None else self._last_transition.as_dict()

    @staticmethod
    def get_transition_schema():
        """Return an independent dictionary of all transition fields."""
        from peacsim.transition_record import get_transition_schema
        return get_transition_schema()

    def get_job_array(self):
        """Return an owned task-array snapshot and its simulation time in seconds.

        time_seconds is None before the t=0 cycle, then the last completed
        post-scheduling sample time. Query between cycles on the same thread.
        The array includes the simulator's full task table, including future
        arrivals; this diagnostic interface is not an agent observation filter.
        """
        self._require_cycle_boundary()
        time_seconds = (None if self._next_cycle == 0 else
                        (self._next_cycle - 1) * self._time_granularity)
        return {'time_seconds': time_seconds, 'job_table': self._job_table.copy()}

    def get_job_report(self):
        """Return an owned copy of the successfully finalized task report."""
        if not self._run_finalized:
            raise RuntimeError('Task report is unavailable until finalize() succeeds.')
        return self._job_report.copy(deep=True)

    def get_tracking_report(self):
        """Owned formal IT tracking summary, available after successful finalize.

        Its artifact_path identifies full-precision replay inputs. The legacy
        tracking_error_at_90 attribute and run() cost tuple keep their old meaning.
        """
        if not self._run_finalized:
            raise RuntimeError('Tracking report is unavailable until finalize() succeeds.')
        return deepcopy(self._tracking_report)

    def get_energy_cost_report(self):
        """Owned IT energy/price estimate, separate from performance and legacy costs.

        status and the covered window distinguish partial native observations.
        No configured price means amount=None, not a zero electricity bill.
        """
        if not self._run_finalized:
            raise RuntimeError('Energy/cost report is unavailable until finalize() succeeds.')
        return deepcopy(self._energy_cost_report)

    @staticmethod
    def _make_job_report(table):
        """Convert a private copy, never sharing the live array's storage."""
        report = pd.DataFrame(table.T.copy(), columns=JOB_COLUMNS)
        report['job_id'] = report['job_id'].astype(int)
        report['job_type_id'] = report['job_type_id'].astype(int)
        return report

    def update_cluster(self, current_time):
        """
        Marks servers as idle if they have finished their jobs. Updates the job table with the current time.
        Calculates several variables used for the output file.
        """
        finished_servers, finished_jobs = self.check_all_other_servers_finished()
        if getattr(self, '_transition_capture', None) is not None:
            self._transition_capture.capture_completion(self._node_table, finished_servers, finished_jobs)
        if finished_servers.size > 0 or finished_jobs.size > 0:
            self._node_table[server_job_id, finished_servers] = self._idle_status
            self._node_table[server_state, finished_servers] = self._idle_status
            self._node_table[server_power, finished_servers] = self._idle_power

            # Old sim behavior
            # self._node_table[job_progress, finished_servers] = 0

            self._job_table[end_time, finished_jobs] = current_time
            self._job_table[estimate_finished, finished_jobs] = 1
            self._job_table[update_time, finished_jobs] = current_time

        self._node_table[estimate_power] = self._node_table[server_power]
        self._node_table[just_started] = -1
        self._node_table[control_power] = -1

        running_mask = self._node_table[server_state] != self._idle_status
        running_job_ids = self._node_table[server_job_id, running_mask].astype(int)
        job_progress_values = self._node_table[job_progress, running_mask]
        self._job_table[estimate_finished, running_job_ids] = job_progress_values
        self._job_table[update_time, running_job_ids] = current_time

        # Output variables
        self.standby_power = 0

        if self._runtime_policy.availability_qos:
            waiting_mask = (self._job_table[start_time] == self._idle_status) & (
                    self._job_table[arrival_time] <= current_time)
            waiting_jobs = self._job_table[:, waiting_mask]
            job_type_ids, waiting_counts = np.unique(waiting_jobs[job_type_id], return_counts=True)

            self.waiting_jobs_by_job_type.append(waiting_counts)
        self.waiting_sum = len(np.where(
            (self._job_table[start_time] == self._idle_status) & (self._job_table[arrival_time] <= current_time))[0])
        self.server_state_at_update = self._node_table[server_state]
        if current_time >= self._iso_signal_granularity:
            last_power = self._cluster_power_cap_per_signal[current_time // self._iso_signal_granularity - 1]
            self.tracking_error = (self.real_power - last_power) / (self._runtime_policy.R + 0.0001)
        else:
            self.tracking_error = 0

    def check_all_other_servers_finished(self):
        """Return completed active nodes/jobs; reject partial group completion.

        Native multi-node jobs run synchronously. Check before update_cluster
        records completion or releases any node; asynchronous waiting/release
        semantics are intentionally undefined. Idle-node progress is stale data.
        """
        nodes = self._node_table
        active = nodes[server_state] != -1
        finished = active & (nodes[job_progress] >= 1)
        finished_jobs = np.unique(nodes[server_job_id, finished]).astype(int)
        if finished_jobs.size:
            unfinished_peers = active & ~finished & np.isin(nodes[server_job_id], finished_jobs)
            if unfinished_peers.any():
                partial_job = int(nodes[server_job_id, unfinished_peers][0])
                peers = active & (nodes[server_job_id] == partial_job)
                done = nodes[node_id, peers & finished].astype(int).tolist()
                pending = nodes[node_id, peers & ~finished].astype(int).tolist()
                raise RuntimeError(
                    f'Unsupported asynchronous multi-node execution for job {partial_job}: '
                    f'finished nodes {done}, unfinished nodes {pending}; '
                    'nodes of the same job must finish together.')
        return nodes[node_id, finished].astype(int), finished_jobs

    def update_job_progress(self, current_time):
        active_mask = (self._node_table[server_state] != self._idle_status) & (self._node_table[just_started] != 1)
        idle_mask = ~active_mask
        just_started_mask = self._node_table[just_started] == 1
        control_power_mask = self._node_table[control_power] != -1

        active_job_progress = self._time_granularity * calculate_unit_progress(self._node_table[:, active_mask])
        self._node_table[job_progress, active_mask] += active_job_progress

        self._node_table[server_power, control_power_mask] = self._node_table[control_power, control_power_mask]

        if idle_mask.any():
            self._node_table[job_progress, idle_mask] = 0

        if just_started_mask.any():
            self._node_table[job_progress, just_started_mask] = 0

        if getattr(self, '_transition_capture', None) is not None:
            self._transition_capture.capture_progress(
                self._node_table, active_mask, just_started_mask, control_power_mask, active_job_progress)

    def calc_tracking_num(self, p_target):
        """
        Calculates tracking_num for output file
        """
        real_delta_power = p_target - np.sum(self._node_table[server_power])
        increment = (sum(self._job_config.all_max_job_power.values()) /
                     float(self._job_type_count) - self._idle_power)
        # A legacy diagnostic, not an actuator or the formal tracking metric.
        # Zero W/node cannot convert a power gap to a node count. Keep it missing
        # rather than inventing a finite count or blocking a valid fixed budget.
        self.tracking_num = None if increment == 0 else int(math.ceil(real_delta_power / increment))

    def update_realtime_qos(self, t):
        """
        realtime_qos = (waitingTime + elapsedTime + minRemainingTime - minExecutionTime) / minExecutionTime
        Also sets update_time to current_time for all jobs that have not started but have arrived.
        """
        self._job_table[update_time, np.where((self._job_table[start_time] == self._idle_status)
                                              & (self._job_table[arrival_time] < t))[0]] = t
        self._job_table[realtime_qos] = ((self._job_table[update_time] -
                                         self._job_table[arrival_time]) / self._job_table[min_execution_time] -
                                         self._job_table[estimate_finished])

    def calc_qos_emergency(self, current_time):
        """
        Sets qos_emergency to True if any job has passed its deadline. Otherwise, sets it to False.
        """
        queued_jobs = np.where(
            (self._job_table[start_time] == self._idle_status) & (self._job_table[arrival_time] <= current_time))[0]
        boundary_start_time = (self._job_table[arrival_time, queued_jobs] +
                               np.multiply(self._job_table[qos_constraint, queued_jobs],
                                           self._job_table[min_execution_time, queued_jobs]))
        start_emergency = (current_time >= boundary_start_time).any()
        if start_emergency:
            emergency_idx = np.argmax(current_time >= boundary_start_time)
            emergency_end = current_time + self._job_table[min_execution_time, queued_jobs][emergency_idx]
        else:
            emergency_end = -1
        self.qos_emergency = start_emergency or (current_time <= emergency_end)

    def record_state(self, t, qos_emergency, real_power, p_target, tracking_num, waiting_sum, tracking_error,
                     qos_job_sum, standby_power, f):
        """
        Writes the state of the simulator to the output file
        """
        f.write('%d,%d,%s,%d,%.1f,%.1f,%.3f,%.1f,%.1f,' % (
            t, int(qos_emergency), '' if tracking_num is None else '%d' % tracking_num,
            waiting_sum, p_target, real_power,
            tracking_error, qos_job_sum, standby_power))
        f.write(','.join([str(x) for x in self._node_table[server_power]]))
        sum_estimate_power = np.sum(self._node_table[estimate_power])
        f.write(',%.1f,' % sum_estimate_power)
        # f.write(','.join([str(x) for x in self.server_state_at_update]))
        f.write(','.join([str(x) for x in self._node_table[server_state]]))
        f.write('\n')

    def _close_trace(self, output_stream):
        """Invalidate this episode if closing its owned trace fails to flush."""
        try:
            output_stream.close()
        except BaseException:
            self._cycle_failed = True
            raise

    def _limit_current_it(self, limit_w):
        """Apply an optional immediate safety limit before progress and metering."""
        nodes = self._node_table
        active = nodes[server_state] != self._idle_status
        headroom = nodes[server_power, active] - nodes[min_job_power, active]
        excess = float(np.sum(nodes[server_power])) - limit_w
        reducible = float(np.sum(headroom))
        if excess > 0 and reducible > 0:
            nodes[server_power, active] -= min(1., excess / reducible) * headroom

    def advance_cycle(self, output_stream=None, *, it_power_budget_w=None, task_weights=None,
                      immediate_it_limit_w=None, priority_types=None):
        """Execute one original loop iteration, returning True if one was run.

        The first call processes t=0 (no elapsed interval); later calls process
        multiples of time_granularity, including the original last sample.
        After that sample, return False without touching state, output or RNG.
        Scheduling still runs once per cycle with the original actuator delay.
        An optional external IT budget replaces only this cycle's policy input.
        Supply it before advancing; it is also the requested target for the
        elapsed interval, not a claim about its actual power. None preserves
        the original ISO-derived target. No budget is implicitly held here.
        Optional task weights apply only to this cycle's scheduling call and
        its weighted power estimate; omission uses the original policy weights.

        As with run(), callers must first initialize output headers. If no stream
        is supplied, append one row to power_trace.csv and close the file before
        returning. A supplied stream is borrowed and is not closed or flushed.
        Seed the existing global RNGs only at the first cycle. Interleaving other
        RNG users is not isolated. On an exception, discard this instance; a
        partially executed cycle cannot be retried safely.

        This method does not finalize reports or reset an episode. Once all
        cycles are processed, call finalize(); run() also finishes remaining
        cycles and then finalizes. Flush/close any borrowed trace stream before
        finalization, which reads power_trace.csv for the existing metrics.
        """
        self._require_cycle_boundary()
        if self._next_cycle > self._simulation_duration // self._time_granularity:
            return False
        if it_power_budget_w is not None:
            from peacsim.metrics.tracking import _number
            it_power_budget_w = _number(it_power_budget_w, 'it_power_budget_w', True)
        if immediate_it_limit_w is not None:
            from peacsim.metrics.tracking import _number
            immediate_it_limit_w = _number(immediate_it_limit_w, 'immediate_it_limit_w', True)
        if task_weights is not None:
            if not hasattr(self._runtime_policy, 'execute_with_weights'):
                raise ValueError('this policy does not support external task weights')
            task_weights = self._runtime_policy.validate_task_weights(task_weights)

        owns_stream = output_stream is None
        if owns_stream:
            output_stream = open(self._output_dir + 'power_trace.csv', 'a')
        self._cycle_in_progress = True
        try:
            if self._next_cycle == 0:
                np.random.seed(self._random_seed)
                random.seed(self._random_seed)

            current_time = self._next_cycle * self._time_granularity
            p_target = self._cluster_power_cap_per_signal[current_time // self._iso_signal_granularity]
            if it_power_budget_w is not None:
                p_target = it_power_budget_w
            if immediate_it_limit_w is not None:
                self._limit_current_it(immediate_it_limit_w)
            self._transition_capture = TransitionRecord(self, current_time, p_target)
            if current_time > 0:
                self.update_job_progress(current_time)
            self.update_cluster(current_time)
            self._transition_capture.capture_after_release(self._node_table)
            self.update_realtime_qos(current_time)
            self.calc_qos_emergency(current_time)
            if priority_types is not None:
                jobs = self._job_table
                mask = (np.isin(jobs[job_type_id], priority_types) & (jobs[arrival_time] <= current_time)
                        & (jobs[end_time] == self._idle_status))
                self._runtime_policy.priority_jobs = frozenset(jobs[job_id, mask].astype(int))
            if task_weights is None:
                self._runtime_policy.execute(current_time, p_target)
            else:
                self._runtime_policy.execute_with_weights(current_time, p_target, task_weights)
            if self._qos_boundary_snapshot is None and current_time >= self._qos_cohort_end:
                self._qos_boundary_snapshot = (current_time, self._job_table.copy())
            # self.probabilistic_results.append(self._runtime_policy.probabilistic_results)
            self.calc_tracking_num(p_target)
            self.real_power = np.sum(self._node_table[server_power])
            self.qos_job_sum = self.real_power
            self.record_state(current_time, int(self.qos_emergency), self.real_power, p_target,
                              self.tracking_num, self.waiting_sum, self.tracking_error, self.qos_job_sum,
                              self.standby_power, output_stream)
            self._transition_capture.finish(self._node_table)
            self._tracking_recorder.append(self._transition_capture.tracking_input(), self._episode_id,
                                           external_it_budget_w=it_power_budget_w)
            self._next_cycle += 1
        except BaseException:
            self._cycle_failed = True
            raise
        finally:
            self._cycle_in_progress = False
            transition, self._transition_capture = self._transition_capture, None
            if owns_stream:
                self._close_trace(output_stream)
        # Publish only after this call's owned trace has closed successfully.
        self._last_transition = transition
        self._last_policy_target_w = float(p_target)
        return True

    def run(self):
        """Finish remaining cycles, then perform the original final export once."""
        self._require_cycle_boundary()
        if self._run_finalized:
            raise RuntimeError('Simulation already finalized; create a new Simulator.')
        print("---> EXECUTING SIMULATOR <---")
        print("Job Table:", self._job_table.shape)
        print("P: ", self._runtime_policy.P, "R: ",  self._runtime_policy.R)
        print("Job weights", self._runtime_policy.weights)
        print("Random seed:", self._random_seed)
        print("Simulation duration (s):", self._simulation_duration)

        outputStream = open(self._output_dir + 'power_trace.csv', 'a')
        try:
            for _ in tqdm(range(self._next_cycle, self._simulation_duration // self._time_granularity + 1),
                          desc="Simulation Progress"):
                self.advance_cycle(outputStream)
        finally:
            self._close_trace(outputStream)
        return self.finalize()

    def finalize(self):
        """Finalize a completed simulation without advancing or changing physics.

        Return the original (power_cost, tracking_cost) pair. Repeated successful
        calls return the cached pair without reading/writing files or using RNG.
        An unfinished simulation is rejected. On an export failure, keep reports
        unpublished; partial output files may remain. After fixing the I/O cause,
        retrying overwrites derived reports without appending any trace samples.
        """
        self._require_cycle_boundary()
        if self._run_finalized:
            return self._final_costs
        if self._next_cycle <= self._simulation_duration // self._time_granularity:
            raise RuntimeError('Simulation has not reached its final sample; continue advancing first.')

        report = self._make_job_report(self._job_table)
        previous_summary = self.qos_summary
        try:
            tracking_error_at_90 = self.error_violate()
            if self._qos_boundary_snapshot is not None:
                observed, table = self._qos_boundary_snapshot
                snapshot = self._make_job_report(table)
                write_qos_summary_df(self, snapshot, observed, suffix='_at_cohort_end')
            violations, violation_ratio = write_qos_summary_df(self, report)
            costs = self._write_final_output(report)
            tracking_report = self._tracking_recorder.publish(
                self._output_dir, self._episode_id, self._random_seed, it_tariff=self._it_tariff)
        except Exception:
            self.qos_summary = previous_summary
            raise

        self.tracking_error_at_90 = tracking_error_at_90
        self.constraint_violations = violations
        self.ratio_of_job_type_qos_violation = violation_ratio
        self._job_report = report
        self._final_costs = costs
        self._tracking_report = tracking_report
        self._energy_cost_report = self._tracking_recorder.energy_cost_report
        self._tracking_recorder = None  # Full history is in the artifact; retain only the summary.
        self._run_finalized = True
        print('Formal IT tracking [{}]: {}'.format(tracking_report['status'], tracking_report['artifact_path']))
        return costs

    def error_violate(self):
        """
        Checking whether tracking error is violated. Uses output csv file to check for this.
        """
        # get the percentage of error < 0.3.
        power_trace_df = pd.read_csv(self._output_dir + 'power_trace.csv')
        error_at_90 = get_write_tracking_error_90th(power_trace_df=power_trace_df, output_dir=self._output_dir)
        print('Legacy CSV tracking error at 90%% percentile: %.4f' % error_at_90)
        if error_at_90 > 0.3:
            print('Tracking error violated.')
        return error_at_90

    def qos_violate(self, time_length):
        """Use the same cohort and deadline rule as the final report and objective."""
        from peacsim.qos_evaluation import evaluate_qos, constraint_results
        _, summary = evaluate_qos(
            self.get_job_report(), self._job_type_count,
            self._job_config.all_job_qos_constraints, self._job_config.all_min_execution_time,
            time_length, self._qos_cohort_start, self._qos_cohort_end)
        return constraint_results(summary, 1 - self.qos_violation_percentile_cutoff)

    def final_output(self):
        """Compatibility entry: use the guarded, idempotent finalization path."""
        return self.finalize()

    def _write_final_output(self, job_report):
        hour = 3600
        kwh = hour * 1000
        t_low = 0
        t_high = self._simulation_duration
        h_low = 0
        h_high = 1 * self._simulation_duration // hour
        job_report['sys_time'] = job_report['end_time'] - job_report['arrival_time']
        job_report['execution_time'] = job_report['end_time'] - job_report['start_time']
        qos_deg_avg = []
        for job_type in range(self._job_type_count):
            this_type = job_report[job_report['job_type_id'] == job_type]
            this_type_in_range = this_type[((this_type['arrival_time'] >= t_low)
                                            & (this_type['arrival_time'] <= t_high))
                                           | ((this_type['end_time'] >= t_low) & (this_type['end_time'] <= t_high))]
            if len(this_type_in_range) == 0:
                qos_deg_avg.append(0)
            else:
                qos_deg_avg.append(this_type_in_range['realtime_qos'].mean())
        with open(self._output_dir + 'QoSDegAvg.csv', 'w') as f:
            f.write('jobTypeID,QoSDegAvg\n')
            for job_type, qos_deg_avg in enumerate(qos_deg_avg):
                f.write('%d,%.2f\n' % (job_type, qos_deg_avg))
        df = pd.read_csv(self._output_dir + 'power_trace.csv', usecols=['time', 'trackingError'])
        pr_table = pd.read_csv(self._output_dir + 'PRtable.csv', index_col='phase')

        piP, piR, piE = 0.1 / kwh, 0.1 / kwh, 0.1 / kwh
        cost = 0

        power_cost = 0
        tracking_cost = 0

        for phase in range(h_low, h_high):  ### for multi hour experiments
            err_phase = df[(df['time'] >= phase * hour) & (df['time'] <= (phase + 1) * hour)]['trackingError'][1:]
            abs_err_phase = [abs(float(x)) for x in err_phase.values]
            # P, R = pr_table.at[phase, 'P'], pr_table.at[phase, 'R']

            P, R = self._runtime_policy.P, self._runtime_policy.R
            power_cost += (piP * P - piR * R) * hour
            tracking_cost += piE * R * sum(abs_err_phase) / len(abs_err_phase) * hour

            cost += (piP * P - piR * R) * hour + piE * R * sum(abs_err_phase) / len(abs_err_phase) * hour

        # WARNING: THIS PART IS ONLY RELEVANT TO RSR, commenting out the parameterize cost
        # calculations in calculate_monetary_cost file

        print('Cost RSR:', cost)

        cost_df = pd.DataFrame({'cost': [cost]})
        # Save the DataFrame to a CSV file
        cost_df.to_csv(self._output_dir + 'cost_power_rsr.csv', index=False, header=False)

        job_report.to_csv(self._output_dir + 'job_table.csv', index=False)
        print("Results saved in:", self._output_dir)
        if self._runtime_policy.availability_qos:
            pd.DataFrame(self.waiting_jobs_by_job_type).to_csv(self._output_dir + 'waiting_jobs_by_job_type.csv', index=False)
            pd.DataFrame(self.probabilistic_results).to_csv(self._output_dir + 'probabilistic_results.csv', index=False)
        return power_cost, tracking_cost
