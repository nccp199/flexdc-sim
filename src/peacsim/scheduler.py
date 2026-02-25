from peacsim.node_table_dao import *
from peacsim.job_table_dao import *
from peacsim.macros import *
import random
import math

def assign_idle_servers(node_table, job_table, current_time, 
                        total_num_of_active_servers, idle_server_ids, job_queue,
                        server_count, job_config, job_type_count, weights):
    """
    Assigns idle servers in node_table to jobs waiting in their queues. The order in which we assign jobs from
    a given queue is FIFO (first in first out). The order in which we pick queues to run is randomized to avoid bias
    Updates node_table when an idle server becomes active, and updates job_table when a job moves from the queue to
    a server.
    """
    # this variable stands for the total number of servers that can be active with the given power target

    num_servers_activated = max(0, total_num_of_active_servers - (server_count - len(idle_server_ids)))

    adjusted_weights = adjust_weights(current_time, job_table, weights, job_queue)

    server_count_change_per_job_type = \
        calculate_server_count_change(node_table, total_num_of_active_servers, adjusted_weights, job_type_count)

    job_order = list(range(job_type_count))
    random.shuffle(job_order)

    index = 0
    for job_type in job_order:
        job_size = job_config.all_job_size[job_type]
        job_type_queue = job_queue[0, np.where(job_queue[1] == job_type)[0]]
        # active_servers_for_job_type should be the number of servers that are currently running the job_type
        active_servers_for_job_type = len(np.where(node_table[server_state] == job_type)[0])
        # num_to_be_activated = round((server_count_change[job_type] - active_servers_for_job_type) / job_size) * job_size
        num_to_be_activated = round((server_count_change_per_job_type[job_type]) / job_size) * job_size
        num_to_be_scheduled = max(0, num_to_be_activated)

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

        job_table[start_time, job_ids] = current_time  # start_time

        # update node_table
        max_power = job_config.all_max_job_power[job_type]
        node_table[server_job_id, server_indices] = jobs_ids_repeated
        node_table[server_state, server_indices] = job_type
        node_table[estimate_power, server_indices] = max_power
        node_table[min_exec_time, server_indices] = job_config.all_min_execution_time[job_type]
        node_table[max_exec_time, server_indices] = job_config.all_max_execution_time[job_type]
        node_table[min_job_power, server_indices] = job_config.all_min_job_power[job_type]
        node_table[max_job_power, server_indices] = max_power
        node_table[just_started, server_indices] = 1
        node_table[server_power, server_indices] = max_power

        index += num_servers_scheduled
        num_servers_activated -= num_servers_scheduled

def adjust_weights(current_time, job_table, weights, job_queue):
    """
    Normalizes weights for job queues that are not empty by replacing weights with zero for empty queues.
    Returns a dictionary with the active weight and the associated job_type.
    """
    queued_job_types = np.unique(job_table[job_type_id, (job_table[start_time] == JOB_NOT_STARTED) &  ### TODO use job queue instead
                                                              (job_table[arrival_time] <= current_time)])

    adjusted_weights = {job_type: weight if job_type in queued_job_types else 0 for job_type, weight in
                        weights.items()}

    # Normalize the adjusted weights
    total_active_weights = sum(adjusted_weights.values())
    if total_active_weights > 0:
        adjusted_weights = {job_type: weight / total_active_weights for job_type, weight in
                            adjusted_weights.items()}
    return adjusted_weights

def calculate_server_count_change(node_table, total_num_of_active_servers, adjusted_weights, job_type_count):
    servers_to_be_active = [max(0, round(weight * total_num_of_active_servers))
                            for weight in adjusted_weights.values()]

    # Filter out idle servers (-1)
    non_idle_job_type_ids = node_table[server_state][node_table[server_state] != SERVER_STATUS_IDLE] \
        .astype(int)

    # Count occurrences of each job_type_id
    max_job_type_id = job_type_count  # Total number of job types
    job_type_counts = np.bincount(non_idle_job_type_ids, minlength=max_job_type_id)

    server_count_change = np.maximum(np.asarray(servers_to_be_active) - job_type_counts, 0)

    return server_count_change.tolist()


