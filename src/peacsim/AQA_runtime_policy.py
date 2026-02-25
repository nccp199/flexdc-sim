import math
from .runtime_policy import RuntimePolicy
import numpy as np
import pandas as pd
import random

(node_id, server_job_id, server_state, server_power, estimate_power, job_progress, min_exec_time, max_exec_time,
 min_job_power, max_job_power, just_started, control_power) = range(12)

(job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time, realtime_qos,
 min_execution_time, qos_constraint) = range(10)


class AQARuntimePolicy(RuntimePolicy):
    def __init__(self, policy_config, exp_config, job_config, node_table, job_table):
        super().__init__(policy_config, exp_config, job_config, node_table, job_table)
        df = pd.read_csv(self._policy_parameters['weights_path'])
        self.idle_status = -1
        self.weights = {index: float(row.iloc[1]) for index, row in df.iterrows()}
        self.weighted_average_power = sum(
            self.weights[job_type] * self._job_config.all_max_job_power[job_type] for job_type in self.weights)
        self.job_type_count = job_config.job_type_count
        self.availability_qos = False
        if policy_config.runtime_policy_name == 'flex-resource':
            self.probabilities = policy_config.policy_parameters['probabilities']
            self.availability_qos = True
            print(self.probabilities)
            self.probabilistic_results = []
        print("Policy: ", self._runtime_policy_name)
        print("Availability QoS: ", self.availability_qos)
        # self.applied_power_cap = [0] * (exp_config.simulation_duration + 1)

    def execute(self, current_time, p_target):
        """
        This function is the main function for the AQA runtime policy. It assigns idle servers to jobs in their
        respective queues. After this, it adjusts the power caps if needed. Also sets estimate_power to max_job_power
        for all active servers.
        """
        self._node_table[estimate_power, np.where(self._node_table[server_state] != self.idle_status)[0]] = \
            [self._node_table[max_job_power, np.where(self._node_table[server_state] != self.idle_status)[0]]]
        self.assign_idle_servers(current_time, p_target)
        delta_power = p_target - np.sum(self._node_table[estimate_power])
        if delta_power < 0:
            # self.applied_power_cap[current_time] = abs(delta_power)
            self.adjust_server_power(abs(delta_power), current_time)

    def adjust_server_power(self, delta_power, t=0):
        """
        This function adjusts the power caps of active servers. The power cap is adjusted by reducing the power cap of
        each server by a fraction of the total reducible power. The fraction is calculated by dividing the delta power
        by the total reducible power. The power caps are applied to EVERY active server (regardless of job type). Power
        caps are only adjusted in the case where power target is less than the sum of pjmax, which is the power of a
        server without any caps applied to it.
        """
        active_indices = np.where(self._node_table[server_state] != self.idle_status)[0]

        max_reducible_power = np.sum(
            self._node_table[max_job_power, active_indices] - self._node_table[min_job_power, active_indices])

        if len(active_indices) > 0 and max_reducible_power > 0:

            if self._runtime_policy_name  == 'priority-cap':
                # 1. Create a lookup map for QoS constraints from the job table {job_id: qos_constraint}.
                qos_map = dict(zip(self._job_table[job_id, :].astype(int), self._job_table[qos_constraint, :]))

                # 2. Get the QoS constraint for each job running on an active server.
                active_job_ids = self._node_table[server_job_id, active_indices].astype(int)
                active_qos = np.array([qos_map.get(j_id, 0) for j_id in active_job_ids])

                # 3. Get the unique QoS priority levels of active jobs and sort them from highest to lowest.
                # This determines the order in which we will cap the groups.
                priority_levels = np.sort(np.unique(active_qos))[::-1]

                # 4. Start with all servers at their maximum power; we will reduce them in priority groups.
                new_power = self._node_table[max_job_power, active_indices].copy()
                power_to_reduce = abs(delta_power)

                # 5. Iterate through each priority level, starting with the one with the loosest QoS (highest value).
                for qos_level in priority_levels:
                    if power_to_reduce <= 0:
                        break

                    # Identify all active servers belonging to the current priority group.
                    group_mask = (active_qos == qos_level)

                    # Get the power properties for just this group.
                    p_max_group = self._node_table[max_job_power, active_indices[group_mask]]
                    p_min_group = self._node_table[min_job_power, active_indices[group_mask]]

                    # Calculate the maximum possible power reduction for each server in the group (50% of its range).
                    max_reducible_per_server = (p_max_group - p_min_group) * float(self._policy_parameters['power_cap_percentage'])
                    total_group_reducible = np.sum(max_reducible_per_server)

                    if total_group_reducible <= 0:
                        continue

                    # Determine the reduction to apply to this group.
                    if power_to_reduce >= total_group_reducible:
                        # If we need to reduce more power than this entire group can provide,
                        # cap every server in this group by its maximum allowed amount.
                        reduction_per_server = max_reducible_per_server
                        power_to_reduce -= total_group_reducible
                    else:
                        # If this group can cover the remaining reduction, apply a fractional cap.
                        # This ensures the reduction is distributed proportionally and EQUALLY
                        # across all servers in the group based on their individual capacities.
                        fraction = power_to_reduce / total_group_reducible
                        reduction_per_server = max_reducible_per_server * fraction
                        power_to_reduce = 0

                    # Apply the calculated reduction to all servers in the current priority group.
                    new_power[group_mask] -= reduction_per_server

            else: ## AQA, proportional-cap, and flex-resource
                reduction_fraction = min(1, abs(delta_power) / max_reducible_power)

                new_power = self._node_table[max_job_power, active_indices] - reduction_fraction * (
                        self._node_table[max_job_power, active_indices] - self._node_table[min_job_power, active_indices])

            self._node_table[control_power, active_indices] = new_power
            self._node_table[estimate_power, active_indices] = new_power

    def probabilistic_unlimited_capacity(self):
        # Generate a random TRUE or FALSE for each job type
        is_job_full_availability = [np.random.rand() < prob for prob in self.probabilities]
        return is_job_full_availability

    def assign_idle_servers(self, current_time, p_target):
        """
        Assigns idle servers in node_table to jobs waiting in their queues. The order in which we assign jobs from
        a given queue is FIFO (first in first out). The order in which we pick queues to run is randomized to avoid bias
        Updates node_table when an idle server becomes active, and updates job_table when a job moves from the queue to
        a server.
        """
        idle_server_ids = self._node_table[node_id, np.where(self._node_table[server_state] == self.idle_status)[0]].astype(int)

        job_queue = self._job_table[0:2, np.where((self._job_table[start_time] == self.idle_status)  # TODO: Replace the slicing with magic numbers 0:2
                                                  & (self._job_table[arrival_time] <= current_time))[0]]
        if job_queue.size == 0 or len(idle_server_ids) == 0:
            return

        #### OLD CODE
        # num_servers_activated = max(0, self.calculate_total_active_servers(p_target)
        #                             - (self._server_count - len(idle_server_ids)))
        # server_count_change_full_avail_per_job_type = self.calculate_server_count_change(current_time)

        # this variable stands for the total number of servers that can be active with the given power target
        total_num_of_active_servers = self.calculate_total_active_servers(p_target)
        num_servers_activated = max(0, total_num_of_active_servers - (self._server_count - len(idle_server_ids)))

        adjusted_weights = self.weights

        # AQA normally gives the servers of the jobs with empty queues to the other jobs,
        # in availaibility_qos we don't want that to ensure jobs always have servers ready

        if not self.availability_qos:
            adjusted_weights = self.adjust_weights(current_time)

        server_count_change_per_job_type = \
            self.calculate_server_count_change(total_num_of_active_servers, adjusted_weights)

        job_order = list(range(self.job_type_count))
        random.shuffle(job_order)

        index = 0
        for job_type in job_order:
            job_size = self._job_config.all_job_size[job_type]
            job_type_queue = job_queue[0, np.where(job_queue[1] == job_type)[0]]
            # active_servers_for_job_type should be the number of servers that are currently running the job_type
            active_servers_for_job_type = len(np.where(self._node_table[server_state] == job_type)[0])
            # num_to_be_activated = round((server_count_change[job_type] - active_servers_for_job_type) / job_size) * job_size
            num_to_be_activated = round((server_count_change_per_job_type[job_type]) / job_size) * job_size
            num_to_be_scheduled = max(0, num_to_be_activated)
            # If availability QoS grants full availability for this job type at this time,
            # allow scheduling up to its target without being limited by DR-imposed active server cap.
            is_guaranteed = (
                self.availability_qos and
                isinstance(self.probabilistic_results, (list, np.ndarray)) and
                len(self.probabilistic_results) == self.job_type_count and
                bool(self.probabilistic_results[job_type])
            )

            if is_guaranteed:
                num_servers_scheduled = min(len(job_type_queue) * job_size, num_to_be_scheduled)
            else:
                num_servers_scheduled = min(len(job_type_queue) * job_size, num_servers_activated,
                                            num_to_be_scheduled)
            if num_servers_scheduled <= 0:
                continue
            num_servers_used = min(num_servers_scheduled, len(idle_server_ids) - index)
            run_job_num = int(num_servers_used // job_size)
            num_servers_used = run_job_num * job_size
            job_ids = job_type_queue[:int(num_servers_scheduled // job_size)].astype(int)

            # Adjust jobs_to_assign dataframe accordingly by repeating indices for job_size > 1
            jobs_ids_repeated = np.repeat(job_ids, job_size)[:int(num_servers_used)]
            server_indices = idle_server_ids[index:index + num_servers_used]

            self._job_table[start_time, job_ids] = current_time  # start_time

            # update node_table
            max_power = self._job_config.all_max_job_power[job_type]
            self._node_table[server_job_id, server_indices] = jobs_ids_repeated
            self._node_table[server_state, server_indices] = job_type
            self._node_table[estimate_power, server_indices] = max_power
            self._node_table[min_exec_time, server_indices] = self._job_config.all_min_execution_time[job_type]
            self._node_table[max_exec_time, server_indices] = self._job_config.all_max_execution_time[job_type]
            self._node_table[min_job_power, server_indices] = self._job_config.all_min_job_power[job_type]
            self._node_table[max_job_power, server_indices] = max_power
            self._node_table[just_started, server_indices] = 1
            self._node_table[server_power, server_indices] = max_power

            index += num_servers_scheduled
            # Only reduce remaining activation budget for non-guaranteed job types.
            if not is_guaranteed:
                num_servers_activated -= num_servers_scheduled

    def calculate_server_count_change(self, total_num_of_active_servers, adjusted_weights):

        if self.availability_qos:
            is_job_full_availability = self.probabilistic_unlimited_capacity()
            self.probabilistic_results = is_job_full_availability

            servers_to_be_active = [max(0, round(adjusted_weights[job_index] * self._server_count))
                                    if probabilistic_guarantee else max(0, round(adjusted_weights[job_index] * total_num_of_active_servers))
                                    for job_index, probabilistic_guarantee in enumerate(is_job_full_availability)]

        else:
            servers_to_be_active = [max(0, round(weight * total_num_of_active_servers))
                                    for weight in adjusted_weights.values()]

        # Filter out idle servers (-1)
        non_idle_job_type_ids = self._node_table[server_state][self._node_table[server_state] != self.idle_status]\
            .astype(int)


        # Count occurrences of each job_type_id
        max_job_type_id = self.job_type_count  # Total number of job types
        job_type_counts = np.bincount(non_idle_job_type_ids, minlength=max_job_type_id)

        server_count_change = np.maximum(np.asarray(servers_to_be_active) - job_type_counts, 0)

        return server_count_change.tolist()

    def adjust_weights(self, current_time):
        """
        Normalizes weights for job queues that are not empty by replacing weights with zero for empty queues.
        Returns a dictionary with the active weight and the associated job_type.
        """
        queued_job_types = np.unique(self._job_table[job_type_id, (self._job_table[start_time] == self.idle_status) &
                                                                  (self._job_table[arrival_time] <= current_time)])

        adjusted_weights = {job_type: weight if job_type in queued_job_types else 0 for job_type, weight in
                            self.weights.items()}

        # Normalize the adjusted weights
        total_active_weights = sum(adjusted_weights.values())
        if total_active_weights > 0:
            adjusted_weights = {job_type: weight / total_active_weights for job_type, weight in
                                adjusted_weights.items()}
        return adjusted_weights

    def calculate_total_active_servers(self, p_target):
        """
        Calculates the number of servers to be active given a target power at a time t.
        Returns an integer value representing the total number of active servers.
        """

        if self._policy_parameters.get('node_count_control'):
            return min(int(
                math.ceil(
                    (p_target - self._idle_power * self._server_count) / (
                                self.weighted_average_power - self._idle_power)
                )
            ), self._server_count)

        else:
            return self._server_count

