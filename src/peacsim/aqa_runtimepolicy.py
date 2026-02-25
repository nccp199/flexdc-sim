import math
import numpy as np
import pandas as pd
import random
from peacsim.runtime_policy_new import RuntimePolicyNew
from peacsim.macros import *
from peacsim.node_table_dao import *
from peacsim.job_table_dao import *
from peacsim.power_capper import equal_proportional_power_capping

class AQARuntimePolicy(RuntimePolicyNew):
    def __init__(self, policy_config, exp_config, job_config, node_table, job_table):
        super().__init__(policy_config, exp_config, job_config, node_table, job_table)


    def adjust_server_power(self, delta_power, max_reducible_power, active_indices):
        """
        This function adjusts the power caps of active servers. The power cap is adjusted by reducing the power cap of
        each server by a fraction of the total reducible power. The fraction is calculated by dividing the delta power
        by the total reducible power. The power caps are applied to EVERY active server (regardless of job type). Power
        caps are only adjusted in the case where power target is less than the sum of pjmax, which is the power of a
        server without any caps applied to it.
        """
        new_power = equal_proportional_power_capping(self._node_table, active_indices, delta_power, max_reducible_power)
        return new_power
