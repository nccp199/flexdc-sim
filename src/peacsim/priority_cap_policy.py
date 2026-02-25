import numpy as np
from peacsim.runtime_policy_new import RuntimePolicyNew
from peacsim.power_capper import priority_power_capping
from peacsim.node_table_dao import *

class PriorityCapPolicy(RuntimePolicyNew):
    def __init__(self, policy_config, exp_config, job_config, node_table, job_table):
        super().__init__(policy_config, exp_config, job_config, node_table, job_table)

    def adjust_server_power(self, delta_power, max_reducible_power, active_indices):
        new_power = priority_power_capping(self._job_table,
                            self._node_table,
                            float(self._policy_parameters['power_cap_percentage']),
                            active_indices,
                            delta_power
                            )
        return new_power
