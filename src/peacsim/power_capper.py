import numpy as np
from peacsim.node_table_dao import *
from peacsim.job_table_dao import *


def equal_proportional_power_capping(node_table, active_indices, delta_power, max_reducible_power):
    reduction_fraction = min(1, abs(delta_power) / max_reducible_power)
    new_power = node_table[max_job_power, active_indices] - reduction_fraction * (
                    node_table[max_job_power, active_indices] - node_table[min_job_power, active_indices])
    return new_power

def priority_power_capping(job_table, node_table, power_cap_percentage, active_indices, delta_power):
    qos_map = dict(zip(job_table[job_id, :].astype(int), job_table[qos_constraint, :]))

    # 2. Get the QoS constraint for each job running on an active server.
    active_job_ids = node_table[server_job_id, active_indices].astype(int)
    active_qos = np.array([qos_map.get(j_id, 0) for j_id in active_job_ids])

    # 3. Get the unique QoS priority levels of active jobs and sort them from highest to lowest.
    # This determines the order in which we will cap the groups.
    priority_levels = np.sort(np.unique(active_qos))[::-1]

    # 4. Start with all servers at their maximum power; we will reduce them in priority groups.
    new_power = node_table[max_job_power, active_indices].copy()
    power_to_reduce = abs(delta_power)

    # 5. Iterate through each priority level, starting with the one with the loosest QoS (highest value).
    for qos_level in priority_levels:
        if power_to_reduce <= 0:
            break

        # Identify all active servers belonging to the current priority group.
        group_mask = (active_qos == qos_level)

        # Get the power properties for just this group.
        p_max_group = node_table[max_job_power, active_indices[group_mask]]
        p_min_group = node_table[min_job_power, active_indices[group_mask]]

        # Calculate the maximum possible power reduction for each server in the group (50% of its range).
        max_reducible_per_server = (p_max_group - p_min_group) * power_cap_percentage
        total_group_reducible = np.sum(max_reducible_per_server)

        if total_group_reducible <= 0:
            continue

        # Determine the reduction to apply to this group.
        if power_to_reduce >= total_group_reducible:
            # If we need to reduce more power than this entire group can provide,
            # cap every server in this group by its maximum allowed amount.
            reduction_per_server = max_reducible_per_server
            power_to_reduce -= total_group_reducible
        else:
            # If this group can cover the remaining reduction, apply a fractional cap.
            # This ensures the reduction is distributed proportionally and EQUALLY
            # across all servers in the group based on their individual capacities.
            fraction = power_to_reduce / total_group_reducible
            reduction_per_server = max_reducible_per_server * fraction
            power_to_reduce = 0

        # Apply the calculated reduction to all servers in the current priority group.
        new_power[group_mask] -= reduction_per_server
    return new_power