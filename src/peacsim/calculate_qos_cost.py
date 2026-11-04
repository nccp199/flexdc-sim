import numpy as np
import pandas as pd

from peacsim.qos_evaluation import evaluate_qos, final_late_rates


def calculate_delay_prob(sim_hour, J, job_table, QoS_constraint, min_execution_time,
                         *, cohort_start=0, cohort_end=None):
    """Exact late/total ratios; pending or absent samples have no scalar score."""
    _, summary = evaluate_qos(job_table, J, QoS_constraint, min_execution_time,
                              sim_hour * 3600, cohort_start, cohort_end)
    return final_late_rates(summary)


def calculate_qos_cost(experiment_config, job_config, policy_config, simulator,
                       fit, fitdown, fitup, sim_hour, J, job_table,
                       QoS_constraint, min_execution_time, cost_function_config, get_each_term=False):

    delay_prob = calculate_delay_prob(
        sim_hour, J, job_table, QoS_constraint, min_execution_time,
        cohort_start=getattr(simulator, '_qos_cohort_start', 0),
        cohort_end=getattr(simulator, '_qos_cohort_end', sim_hour * 3600))

    beta = cost_function_config.gradient_driver_variables['beta']
    rho = cost_function_config.gradient_driver_variables['rho']
    delta = cost_function_config.calculate_gradient_variables['qos_threshold']

    K0 = []  # stores the cost of each job type
    for itype in range(J):
        K0.append(np.log(1 + np.exp(rho * (delay_prob[itype] - delta))))

    print("DELAY PROB", delay_prob)
    if get_each_term:
        return beta * sum(K0), K0

    return beta * sum(K0)


def calculate_qos_cost_sa(sim_hour, J, job_table,
                       QoS_constraint, min_execution_time, cost_function_config, get_each_term=False, *, cohort_start=0, cohort_end=None):

    delay_prob = calculate_delay_prob(sim_hour, J, job_table, QoS_constraint, min_execution_time,
                                      cohort_start=cohort_start, cohort_end=cohort_end)

    beta = cost_function_config.beta
    rho = cost_function_config.rho
    delta = cost_function_config.qos_constraint

    K0 = []  # stores the cost of each job type
    for itype in range(J):
        K0.append(np.log(1 + np.exp(rho * (delay_prob[itype] - delta))))

    print("DELAY PROB", delay_prob)
    if get_each_term:
        return beta * sum(K0), K0

    return beta * sum(K0)


def get_power_trace_df(output_dir):
    power_df = pd.read_csv(output_dir + '/power_trace.csv').iloc[1:]
    state_columns = [col for col in power_df.columns if col.startswith('state')]

    power_df['active_servers'] = (power_df[state_columns] != -1).sum(axis=1)
    power_df['realSum_kw'] = power_df['realSum'] / 1000  # Convert to kw
    power_df['target_kw'] = power_df['target'] / 1000  # Convert to kw

    return power_df


def get_job_table_df(output_dir):
    job_table = pd.read_csv(output_dir + '/job_table.csv')
    return job_table


def calculate_qos_cost_availability(num_servers, weights, resource_allocation_probabilities, output_dir,
                                    cost_function_config):
    # job_table = get_job_table_df(output_dir)
    power_df = get_power_trace_df(output_dir)

    waiting_jobs = pd.read_csv(output_dir + "/waiting_jobs_by_job_type.csv")
    waiting_jobs.fillna(0, inplace=True)
    waiting_jobs.index += 1

    state_columns = [col for col in power_df.columns if col.startswith('state')]
    melted = power_df.melt(id_vars='time', value_vars=state_columns, var_name='server', value_name='job_type')
    # Exclude inactive servers
    active_jobs = melted[melted['job_type'] != -1]
    # Count job types per time
    active_job_type_counts = active_jobs.groupby(['time', 'job_type']).size().unstack(fill_value=0)

    active_job_type_counts.columns = waiting_jobs.columns
    waiting_and_active_jobs_sum = active_job_type_counts.add(waiting_jobs, fill_value=0)

    guaranteed_resources = np.round(num_servers * np.asarray(weights))
    guaranteed_resources.reshape(1, len(guaranteed_resources))
    guaranteed_resources_df = pd.DataFrame([guaranteed_resources], columns=waiting_and_active_jobs_sum.columns)

    is_guarantee_satisfied_per_job_type = waiting_and_active_jobs_sum.ge(guaranteed_resources_df.iloc[0], axis=1)

    # Display the result
    # Count the number of True values in each column
    true_counts = is_guarantee_satisfied_per_job_type.sum()

    # Calculate the percentage of True values in each column
    true_ratio = (true_counts / len(is_guarantee_satisfied_per_job_type))
    cost_per_job_type = resource_allocation_probabilities - true_ratio.values

    abs_cost_per_job_type = abs(resource_allocation_probabilities - true_ratio.values)
    softplus_cost_for_each_job_type = cost_function_config.beta * \
                                      np.log(1 + np.exp(cost_function_config.rho * abs_cost_per_job_type))
    total_cost = np.sum(softplus_cost_for_each_job_type)

    cost_per_job_type_negated = -cost_per_job_type

    # Compute a normalized weight adjustment factor (keeping zero-centered adjustments)
    min_val = np.min(cost_per_job_type_negated)
    max_val = np.max(cost_per_job_type_negated)
    range_val = max_val - min_val

    if range_val > 0:
        normalized_cost_per_job_type = (cost_per_job_type_negated - min_val) / range_val
    else:
        normalized_cost_per_job_type = np.zeros_like(cost_per_job_type_negated)  # Avoid division by zero

    return total_cost, normalized_cost_per_job_type

# def calculate_qos_cost_availability(num_servers, weights, resource_allocation_probabilities, output_dir,
#                                     cost_function_config):
#     # job_table = get_job_table_df(output_dir)
#     power_df = get_power_trace_df(output_dir)
#
#     waiting_jobs = pd.read_csv(output_dir + "/waiting_jobs_by_job_type.csv")
#     waiting_jobs.fillna(0, inplace=True)
#     waiting_jobs.index += 1
#
#     state_columns = [col for col in power_df.columns if col.startswith('state')]
#     melted = power_df.melt(id_vars='time', value_vars=state_columns, var_name='server', value_name='job_type')
#     # Exclude inactive servers
#     active_jobs = melted[melted['job_type'] != -1]
#     # Count job types per time
#     active_job_type_counts = active_jobs.groupby(['time', 'job_type']).size().unstack(fill_value=0)
#
#     active_job_type_counts.columns = waiting_jobs.columns
#     waiting_and_active_jobs_sum = active_job_type_counts.add(waiting_jobs, fill_value=0)
#
#     guaranteed_resources = np.round(num_servers * np.asarray(weights))
#     guaranteed_resources.reshape(1, len(guaranteed_resources))
#     guaranteed_resources_df = pd.DataFrame([guaranteed_resources], columns=waiting_and_active_jobs_sum.columns)
#
#     is_guarantee_satisfied_per_job_type = waiting_and_active_jobs_sum.ge(guaranteed_resources_df.iloc[0], axis=1)
#
#     # Display the result
#     # Count the number of True values in each column
#     true_counts = is_guarantee_satisfied_per_job_type.sum()
#
#     # Calculate the percentage of True values in each column
#     true_ratio = (true_counts / len(is_guarantee_satisfied_per_job_type))
#     cost_per_job_type = abs(resource_allocation_probabilities - true_ratio.values)
#
#     beta = cost_function_config.beta
#     rho = cost_function_config.rho
#     softplus_cost_for_each_job_type = beta * np.log(1 + np.exp(rho * cost_per_job_type))
#     total_cost = np.sum(softplus_cost_for_each_job_type)
#
#     min_val = np.min(softplus_cost_for_each_job_type)
#     max_val = np.max(softplus_cost_for_each_job_type)
#     normalized_cost_per_job_type = (softplus_cost_for_each_job_type - min_val) / (max_val - min_val)
#
#     # Inverse the normalized values
#     inversed_normalized_cost_per_job_type = 1 - normalized_cost_per_job_type
#
#     return total_cost, inversed_normalized_cost_per_job_type
