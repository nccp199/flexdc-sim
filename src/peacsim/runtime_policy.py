import pandas as pd


class RuntimePolicy:
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
        # self._job_type_count = job_config.job_type_count

    def execute(self, current_time, p_target):
        pass

    @property
    def P(self):
        return self._P

    @property
    def R(self):
        return self._R
