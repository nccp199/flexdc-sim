import numpy as np
import pandas as pd

def calculate_delay_prob(sim_hour, J, job_table, QoS_constraint, min_execution_time):
    delay_prob = []

    for ijob in range(J):
        first_hour = 1 if sim_hour > 2 else 0  # Skip hour 0: queues are initially empty. perf is biased positively
        # there. Including hour 0 is better than running a single hour, so only do that if there are 3+ hours.
        last_hour = (
                sim_hour - 1) if sim_hour > 2 else sim_hour  # ignore the final hour if there are enough hours to
        # trim. The last hour has draining jobs which biases our perf positively.
        this_job_finished = job_table[(job_table['job_type_id'] == ijob)
                                      & (job_table['end_time'] != -1)
                                      & (job_table['arrival_time'] > first_hour * 3600)]
        delay = list(this_job_finished['end_time'] - this_job_finished['arrival_time'])  # Sojourn time
        job_should_finish = job_table[(job_table['job_type_id'] == ijob)
                                      & (job_table['arrival_time'] + min_execution_time[ijob] < sim_hour * 3600)
                                      & (job_table['end_time'] == -1)
                                      & (job_table['arrival_time'] > first_hour * 3600)
                                      & (job_table['arrival_time'] <= last_hour * 3600)]
        if len(job_should_finish) > 0:
            delay = delay + [sim_hour * 3600 - x for x in
                             list(job_should_finish['arrival_time'])]  # Use true sim_hour here
            # since we're estimating with start time as "now" at the end of the sim.
        delay = [x - min_execution_time[ijob] for x in delay]  # minus min execution time.
        sort_scaled = np.sort(delay)
        prob = np.linspace(0, 1, len(sort_scaled))  # the CDF.
        # alphas = []
        delay_prob_added = False
        for idx, iprob in enumerate(prob):
            if sort_scaled[idx] > 1 * QoS_constraint[ijob] * min_execution_time[ijob] and not delay_prob_added:
                delay_prob.append(1 - iprob)
                delay_prob_added = True
            if idx == len(prob) - 1 and not delay_prob_added:
                delay_prob.append(0)
                delay_prob_added = True
        if not delay_prob_added:  # no job for this type.
            delay_prob.append(0)
    return delay_prob


def calculate_qos_cost(experiment_config, job_config, policy_config, simulator,
                       fit, fitdown, fitup, sim_hour, J, job_table,
                       QoS_constraint, min_execution_time, cost_function_config, get_each_term=False):

    delay_prob = calculate_delay_prob(sim_hour, J, job_table, QoS_constraint, min_execution_time)

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
                       QoS_constraint, min_execution_time, cost_function_config, get_each_term=False):

    delay_prob = calculate_delay_prob(sim_hour, J, job_table, QoS_constraint, min_execution_time)

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
