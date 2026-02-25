import numpy as np
from peacsim.macros import *

# Columns in job_table
(job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time, realtime_qos,
 min_execution_time, qos_constraint) = range(10)


def get_job_queue(job_table, current_time):
    job_queue = job_table[job_id:arrival_time, np.where
                            ((job_table[start_time] == SERVER_STATUS_IDLE)
                             & (job_table[arrival_time] <= current_time))[0]]

    return job_queue