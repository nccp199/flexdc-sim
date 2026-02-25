from peacsim.power_capper import equal_proportional_power_capping
from peacsim.runtime_policy_new import RuntimePolicyNew
from peacsim.node_table_dao import *

import numpy as np

class ProportionalCapPolicy(RuntimePolicyNew):
    def __init__(self, policy_config, exp_config, job_config, node_table, job_table):
        super().__init__(policy_config, exp_config, job_config, node_table, job_table)

    def adjust_server_power(self, delta_power, max_reducible_power, active_indices):
        new_power = equal_proportional_power_capping(self._node_table,
                            active_indices,
                            delta_power, max_reducible_power
                            )
        return new_power