import math
import numpy as np
import pandas as pd
import os
import random
from tqdm import tqdm

from peacsim.iso_signal import read_iso_signal, calculate_cluster_power_caps
from peacsim.extract_qos_metrics import write_qos_summary_df
from peacsim.extract_power_metrics import get_tracking_error_90th, get_write_tracking_error_90th

"""
Authors: @JoshBard @FatihAcun @EthanL3
Created 02/11/2024
PeacLab
"""

(node_id, server_job_id, server_state, server_power, estimate_power, job_progress, min_exec_time, max_exec_time,
 min_job_power, max_job_power, just_started, control_power) = range(12)

(job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time, realtime_qos,
 min_execution_time, qos_constraint) = range(10)


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
                 experiment_config, job_config, cluster_config, qos_violation_percentile_cut_off=0.9, output_dir=None):

        self.qos_emergency = None
        self.standby_power = None
        self.tracking_num = None
        self.waiting_sum = None
        self.qos_job_sum = None
        self.real_power = None
        self.tracking_error = None
        self.server_state_at_update = None
        self._job_config = job_config
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
        self._utilization = experiment_config.utilization
        self._normalize_iso_signal = experiment_config.normalize_iso_signal
        self._iso_signal_granularity = experiment_config.iso_signal_granularity
        self._idle_power = experiment_config.idle_power

        self._iso_signal = read_iso_signal(experiment_config,
                                           self._simulation_duration,
                                           self._iso_signal_granularity,
                                           self._normalize_iso_signal)

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

    def update_cluster(self, current_time):
        """
        Marks servers as idle if they have finished their jobs. Updates the job table with the current time.
        Calculates several variables used for the output file.
        """
        finished_servers, finished_jobs = self.check_all_other_servers_finished()
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
        """
        Helper function to check if all servers running jobs have finished their jobs. Returns the indices of the
        servers and jobs that have finished.
        """
        return self._node_table[node_id, np.where(self._node_table[job_progress] >= 1)[0]].astype(int), np.unique(self._node_table[server_job_id, np.where(self._node_table[job_progress] >= 1)[0]]).astype(int)

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

    def calc_tracking_num(self, p_target):
        """
        Calculates tracking_num for output file
        """
        real_delta_power = p_target - np.sum(self._node_table[server_power])
        self.tracking_num = int(math.ceil(real_delta_power / (
                sum(self._job_config.all_max_job_power.values()) / float(self._job_type_count) - self._idle_power)))

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
        f.write('%d,%d,%d,%d,%.1f,%.1f,%.3f,%.1f,%.1f,' % (
            t, int(qos_emergency), tracking_num, waiting_sum, p_target, real_power,
            tracking_error, qos_job_sum, standby_power))
        f.write(','.join([str(x) for x in self._node_table[server_power]]))
        sum_estimate_power = np.sum(self._node_table[estimate_power])
        f.write(',%.1f,' % sum_estimate_power)
        # f.write(','.join([str(x) for x in self.server_state_at_update]))
        f.write(','.join([str(x) for x in self._node_table[server_state]]))
        f.write('\n')

    def run(self):
        print("---> EXECUTING SIMULATOR <---")
        print("Job Table:", self._job_table.shape)
        print("P: ", self._runtime_policy.P, "R: ",  self._runtime_policy.R)
        print("Job weights", self._runtime_policy.weights)
        print("Random seed:", self._random_seed)
        print("Simulation duration (s):", self._simulation_duration)

        np.random.seed(self._random_seed)
        random.seed(self._random_seed)

        with open(self._output_dir + 'power_trace.csv', 'a') as outputStream:
            for cycle in tqdm(range(0, self._simulation_duration + 1), desc="Simulation Progress"):
                current_time = cycle * self._time_granularity
                p_target = self._cluster_power_cap_per_signal[current_time // self._iso_signal_granularity]
                if current_time > 0:
                    self.update_job_progress(current_time)
                self.update_cluster(current_time)
                self.update_realtime_qos(current_time)
                self.calc_qos_emergency(current_time)
                self._runtime_policy.execute(current_time, p_target)
                # self.probabilistic_results.append(self._runtime_policy.probabilistic_results)
                self.calc_tracking_num(p_target)
                self.real_power = np.sum(self._node_table[server_power])
                self.qos_job_sum = self.real_power
                self.record_state(current_time, int(self.qos_emergency), self.real_power, p_target,
                                  self.tracking_num, self.waiting_sum, self.tracking_error, self.qos_job_sum,
                                  self.standby_power, outputStream)
        # applied_caps = np.asarray(self._runtime_policy.applied_power_cap)
        # np.savetxt('power_caps.txt', applied_caps)
        self._job_table = pd.DataFrame(self._job_table.T,
                                       columns=['job_id', 'job_type_id', 'arrival_time', 'start_time',
                                                'end_time', 'estimate_finished', 'update_time',
                                                'realtime_qos', 'min_execution_time',
                                                'qos_constraint'])
        self._job_table['job_id'] = self._job_table['job_id'].astype(int)
        self._job_table['job_type_id'] = self._job_table['job_type_id'].astype(int)
        self.tracking_error_at_90 = self.error_violate()
        self.constraint_violations, self.ratio_of_job_type_qos_violation = write_qos_summary_df(self)
        power_cost, tracking_cost = self.final_output()
        return power_cost, tracking_cost

    def error_violate(self):
        """
        Checking whether tracking error is violated. Uses output csv file to check for this.
        """
        # get the percentage of error < 0.3.
        power_trace_df = pd.read_csv(self._output_dir + 'power_trace.csv')
        error_at_90 = get_write_tracking_error_90th(power_trace_df=power_trace_df, output_dir=self._output_dir)
        print('Tracking error at 90%% percentile: %.4f' % error_at_90)
        if error_at_90 > 0.3:
            print('Tracking error violated.')
        return error_at_90

    def qos_violate(self, time_length):
        """
        Checking whether QoS is violated.
        Modified to include un-started and unfinished jobs.
        """
        violate_qos = []
        qos_constraints = self._job_config.all_job_qos_constraints

        started_jobs = self._job_table[
            (self._job_table['end_time'] > self._idle_status) & (self._job_table['arrival_time'] < time_length)]
        if started_jobs.empty:
            print('No started jobs')
            return
        qos = started_jobs.groupby('job_type_id').apply(
            lambda x: (x['end_time'] - x['arrival_time'] - x['min_execution_time']) / x['min_execution_time'])
        str_qos_viol_percentile_cutoff = str(self.qos_violation_percentile_cutoff * 100)
        print('QoS deg at '+ str_qos_viol_percentile_cutoff +' by job type')
        qos_deg_at_percentile = qos.groupby('job_type_id').quantile(self.qos_violation_percentile_cutoff)

        constraint_violations = (qos_deg_at_percentile > pd.Series(list(qos_constraints.values())))
        job_type_qos_violation_count = constraint_violations.sum()
        job_type_qos_violation_ratio = job_type_qos_violation_count / self._job_type_count
        print(qos_deg_at_percentile.to_string())
        # print(f'==> num violations>5: {(q90deg > 5).sum()}')
        print('Mean exec slowdown by job type')
        mean_exec_slowdown = started_jobs.groupby('job_type_id').apply(
            lambda x: ((x['end_time'] - x['start_time'] - x['min_execution_time']) / x['min_execution_time']).mean())
        print(mean_exec_slowdown)
        qos_df = pd.DataFrame({
            'job_type_id': qos_deg_at_percentile.index,
            'qos_' + str_qos_viol_percentile_cutoff + 'th': qos_deg_at_percentile.values,
            'mean_exec_slowdown': mean_exec_slowdown.values,
            'qos_threshold': [qos_constraints[job_type] for job_type in qos_deg_at_percentile.index]
        })
        qos_df.to_csv(self._output_dir + 'qos_summary.csv', index=False, header=True)

        print(f'Count of job types with QoS violations from {self._job_type_count} job types: 'f'{job_type_qos_violation_count}')
        print(f'Ratio of job types with QoS violation: {job_type_qos_violation_ratio}')

        print('Violations:', ', '.join(['' for idx, violation in enumerate(violate_qos) if violation]))
        queued_jobs = self._job_table.loc[(self._job_table['arrival_time'] <= self._simulation_duration)]
        print(f'Ratio of all jobs violating QoS, queued in the first hour (from {len(queued_jobs)} jobs):',
              round((queued_jobs['realtime_qos'].values > queued_jobs['qos_constraint'].values).mean(), 7))

        return constraint_violations, job_type_qos_violation_ratio

    def final_output(self):
        hour = 3600
        kwh = hour * 1000
        t_low = 0
        t_high = self._simulation_duration
        h_low = 0
        h_high = 1 * self._simulation_duration // hour
        self._job_table['sys_time'] = self._job_table['end_time'] - self._job_table['arrival_time']
        self._job_table['execution_time'] = self._job_table['end_time'] - self._job_table['start_time']
        qos_deg_avg = []
        for job_type in range(self._job_type_count):
            this_type = self._job_table[self._job_table['job_type_id'] == job_type]
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

        self._job_table.to_csv(self._output_dir + 'job_table.csv', index=False)
        print("Results saved in:", self._output_dir)
        if self._runtime_policy.availability_qos:
            pd.DataFrame(self.waiting_jobs_by_job_type).to_csv(self._output_dir + 'waiting_jobs_by_job_type.csv', index=False)
            pd.DataFrame(self.probabilistic_results).to_csv(self._output_dir + 'probabilistic_results.csv', index=False)
        return power_cost, tracking_cost
