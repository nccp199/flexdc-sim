import numpy as np


def construct_tables():
    """
    Construct the node_table and job_table for the tests. Uses this ordering from the top of simulator.py:

    (node_id, server_job_id, server_state, server_power, estimate_power, job_progress, min_exec_time, max_exec_time,
    min_job_power, max_job_power, just_started, control_power) = range(12)

    (job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time, realtime_qos,
    min_execution_time, qos_constraint) = range(10)
    """
    node_table = np.array([
        [0.0, 1.0, 2.0],  # node_id
        [0.0, 1.0, 2.0],  # server_job_id
        [-1.0, -1.0, -1.0],  # server_state
        [0.0, 0.0, 0.0],  # server_power
        [0.0, 0.0, 0.0],  # estimate_power
        [0.0, 0.0, 0.0],  # job_progress
        [30.7, 739.0, 52.6],  # min_exec_time
        [35.5, 1211, 70.3],  # max_exec_time
        [234.0, 140.0, 276.0],  # min_job_power
        [276.0, 280.0, 402.0],  # max_job_power
        [0.0, 0.0, 0.0],  # just_started
        [0.0, 0.0, 0.0]  # control_power
    ])

    job_table = np.array([
        [0.0, 1.0, 2.0],  # job_id
        [0.0, 1.0, 2.0, ],  # job_type_id
        [0.0, 0.0, 0.0],  # arrival_time
        [0.0, 0.0, 0.0],  # start_time
        [-1.0, -1.0, -1.0],  # end_time
        [0.0, 0.0, 0.0],  # estimate_finished
        [0.0, 0.0, 0.0],  # update_time
        [0.0, 0.0, 0.0],  # realtime_qos
        [30.7, 739.0, 52.6],  # min_execution_time
        [3.0, 5.0, 4.7]  # qos_constraint
    ])
    return node_table, job_table
