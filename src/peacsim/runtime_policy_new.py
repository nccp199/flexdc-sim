import pandas as pd
import numpy as np
import math
from peacsim.node_table_dao import *
from peacsim.job_table_dao import *
from peacsim.node_table_dao import *
from peacsim import scheduler


class RuntimePolicyNew:
    def __init__(self, policy_config, exp_config, job_config, node_table, job_table):
        self._runtime_policy_name = policy_config.runtime_policy_name
        self._P = policy_config.P
        self._R = policy_config.R
        self._policy_parameters = policy_config.policy_parameters
        self._node_table = node_table
        self._job_table = job_table
        self._idle_power = exp_config.idle_power
        self._job_config = job_config
        self._server_count = exp_config.server_count

        df = pd.read_csv(self._policy_parameters['weights_path'])
        self.weights = {index: float(row.iloc[1]) for index, row in df.iterrows()}
        self.weighted_average_power = sum(
            self.weights[job_type] * self._job_config.all_max_job_power[job_type] for job_type in self.weights)
        self._job_type_count = job_config.job_type_count
        print("Policy: ", self._runtime_policy_name)
        self.availability_qos = False

    @property
    def P(self):
        return self._P

    @property
    def R(self):
        return self._R

    def execute(self, current_time, p_target):
        self._node_table[estimate_power, np.where(self._node_table[server_state] != SERVER_STATUS_IDLE)[0]] = \
            [self._node_table[max_job_power, np.where(self._node_table[server_state] != SERVER_STATUS_IDLE)[0]]]
        
        self.schedule_jobs(current_time, p_target)

        self.apply_power_cap(p_target)
        

    def apply_power_cap(self, p_target):
        delta_power = p_target - np.sum(self._node_table[estimate_power])

        if delta_power < 0:
            active_indices = get_active_servers(self._node_table)
            max_reducible_power = np.sum(
                self._node_table[max_job_power, active_indices] 
                - self._node_table[min_job_power, active_indices]
                )
            if len(active_indices) > 0 and max_reducible_power > 0:
                new_power = self.adjust_server_power(delta_power=abs(delta_power), 
                                                    max_reducible_power=max_reducible_power, 
                                                    active_indices=active_indices)
                self._node_table[control_power, active_indices] = new_power
                self._node_table[estimate_power, active_indices] = new_power

        

    def schedule_jobs(self, current_time, p_target):
        job_queue = get_job_queue(self._job_table, current_time)
        idle_server_ids = get_idle_servers(self._node_table)

        if job_queue.size != 0 and len(idle_server_ids) != 0:
            total_num_of_active_servers = self.calculate_total_active_servers(p_target)
            scheduler.assign_idle_servers(self._node_table,
                                          self._job_table,
                                          current_time,
                                          total_num_of_active_servers,
                                          idle_server_ids,
                                          job_queue,
                                          self._server_count,
                                          self._job_config,
                                          self._job_type_count,
                                          self.weights)



    def adjust_server_power(self, delta_power, max_reducible_power, active_indices):
        pass

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

