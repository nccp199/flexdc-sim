"""
flex_resource_policy.py
Flex Resource Policy - provides probabilistic availability guarantees for job types.

Key differences from AQA:
1. Does NOT adjust weights for empty queues (keeps original weights)
2. Uses probabilistic capacity guarantees per job type
3. Guaranteed job types bypass the DR-imposed server activation limit
4. Uses proportional power capping (same as AQA)
"""

import numpy as np
import random
from peacsim.runtime_policy_new import RuntimePolicyNew
from peacsim.power_capper import equal_proportional_power_capping
from peacsim.node_table_dao import *
from peacsim.job_table_dao import *
from peacsim.macros import *


class FlexResourcePolicy(RuntimePolicyNew):
    def __init__(self, policy_config, exp_config, job_config, node_table, job_table):
        super().__init__(policy_config, exp_config, job_config, node_table, job_table)
        self.probabilities = policy_config.policy_parameters['probabilities']
        self.availability_qos = True
        print("Probabilities:", self.probabilities)
        self.probabilistic_results = []

    def probabilistic_unlimited_capacity(self):
        """
        Generate a random TRUE or FALSE for each job type based on configured probabilities.
        Returns a list of booleans indicating which job types have full availability guarantee.
        """
        is_job_full_availability = [np.random.rand() < prob for prob in self.probabilities]
        return is_job_full_availability

    def adjust_server_power(self, delta_power, max_reducible_power, active_indices):
        """
        Flex-resource uses proportional power capping (same as AQA).
        """
        new_power = equal_proportional_power_capping(
            self._node_table, active_indices, delta_power, max_reducible_power
        )
        return new_power

    def schedule_jobs(self, current_time, p_target):
        """
        Override schedule_jobs to implement availability_qos specific logic:
        1. Don't adjust weights for empty queues
        2. Use probabilistic capacity guarantees
        3. Guaranteed job types bypass activation limits
        """
        job_queue = get_job_queue(self._job_table, current_time)
        idle_server_ids = get_idle_servers(self._node_table)

        if job_queue.size == 0 or len(idle_server_ids) == 0:
            return

        total_num_of_active_servers = self.calculate_total_active_servers(p_target)
        num_servers_activated = max(0, total_num_of_active_servers - (self._server_count - len(idle_server_ids)))

        # Key difference #1: Don't adjust weights for empty queues
        # In availability_qos mode, we keep original weights to ensure servers are always ready
        adjusted_weights = self.weights

        # Calculate server count changes with probabilistic guarantees
        server_count_change_per_job_type = self._calculate_server_count_change_with_availability(
            total_num_of_active_servers, adjusted_weights
        )

        job_order = list(range(self._job_type_count))
        random.shuffle(job_order)

        index = 0
        for job_type in job_order:
            job_size = self._job_config.all_job_size[job_type]
            job_type_queue = job_queue[0, np.where(job_queue[1] == job_type)[0]]

            num_to_be_activated = round((server_count_change_per_job_type[job_type]) / job_size) * job_size
            num_to_be_scheduled = max(0, num_to_be_activated)

            # Key difference #2: Check if this job type has guaranteed availability
            is_guaranteed = (
                isinstance(self.probabilistic_results, (list, np.ndarray)) and
                len(self.probabilistic_results) == self._job_type_count and
                bool(self.probabilistic_results[job_type])
            )

            # Key difference #3: Guaranteed jobs bypass num_servers_activated limit
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

            # Repeat job IDs for multi-server jobs
            jobs_ids_repeated = np.repeat(job_ids, job_size)[:int(num_servers_used)]
            server_indices = idle_server_ids[index:index + num_servers_used]

            # Update job_table
            self._job_table[start_time, job_ids] = current_time

            # Update node_table
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

            # Key difference #4: Only reduce activation budget for non-guaranteed job types
            if not is_guaranteed:
                num_servers_activated -= num_servers_scheduled

    def _calculate_server_count_change_with_availability(self, total_num_of_active_servers, adjusted_weights):
        """
        Calculate server count changes with probabilistic availability guarantees.

        For guaranteed job types: use weight * TOTAL_SERVER_COUNT (full capacity)
        For non-guaranteed: use weight * total_num_of_active_servers (limited by power)
        """
        # Generate probabilistic results for this timestep
        is_job_full_availability = self.probabilistic_unlimited_capacity()
        self.probabilistic_results = is_job_full_availability

        # Calculate target servers per job type
        servers_to_be_active = []
        for job_index, probabilistic_guarantee in enumerate(is_job_full_availability):
            if probabilistic_guarantee:
                # Guaranteed: can use full server capacity
                target = max(0, round(adjusted_weights[job_index] * self._server_count))
            else:
                # Not guaranteed: limited by power target
                target = max(0, round(adjusted_weights[job_index] * total_num_of_active_servers))
            servers_to_be_active.append(target)

        # Count current active servers per job type
        non_idle_job_type_ids = self._node_table[server_state][
            self._node_table[server_state] != SERVER_STATUS_IDLE
        ].astype(int)

        job_type_counts = np.bincount(non_idle_job_type_ids, minlength=self._job_type_count)

        # Calculate how many more servers needed per job type
        server_count_change = np.maximum(np.asarray(servers_to_be_active) - job_type_counts, 0)

        return server_count_change.tolist()