import numpy as np
from peacsim.macros import SERVER_STATUS_IDLE

# Columns in job_table
(node_id, server_job_id, server_state, server_power, estimate_power, job_progress, min_exec_time, max_exec_time,
 min_job_power, max_job_power, just_started, control_power) = range(12)

def get_idle_servers(node_table):
    idle_server_ids = node_table[node_id, np.where(node_table[server_state] == SERVER_STATUS_IDLE)[0]].astype(int)
    return idle_server_ids

def get_active_servers(node_table):
    active_server_ids = np.where(node_table[server_state] != SERVER_STATUS_IDLE)[0].astype(int)
    return active_server_ids

