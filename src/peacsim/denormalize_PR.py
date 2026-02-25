def denormalize_PR(experiment_config, job_config, policy_config):
    avg_max_job_power = sum(job_config.all_max_job_power.values())/len(job_config.all_max_job_power)
    avg_num_servers = experiment_config.server_count * experiment_config.utilization
    avg_idle_servers = experiment_config.server_count - avg_num_servers
    policy_config._P = int(policy_config.P_ratio * (avg_max_job_power * avg_num_servers +
                           experiment_config.idle_power * avg_idle_servers))
    policy_config._R = int(policy_config.R_ratio * (avg_max_job_power * experiment_config.server_count - (experiment_config.idle_power * experiment_config.server_count)) / 2)
    print('P: %d, R: %d' % (policy_config.P, policy_config.R))


def denormalize_p_and_r_to_watts(experiment_config, job_config, p_normalized, r_normalized):
    avg_max_job_power = sum(job_config.all_max_job_power.values())/len(job_config.all_max_job_power)
    avg_num_servers = experiment_config.server_count * experiment_config.utilization
    avg_idle_servers = experiment_config.server_count - avg_num_servers

    P = int(p_normalized * (avg_max_job_power * avg_num_servers +
                            experiment_config.idle_power * avg_idle_servers))
    R = int(r_normalized * (avg_max_job_power * experiment_config.server_count
                            - (experiment_config.idle_power * experiment_config.server_count)) / 2)

    return P, R


def normalize_p_and_r_from_watts(experiment_config, job_config, P, R):
    avg_max_job_power = sum(job_config.all_max_job_power.values()) / len(job_config.all_max_job_power)
    avg_num_servers = experiment_config.server_count * experiment_config.utilization
    avg_idle_servers = experiment_config.server_count - avg_num_servers

    P_normalized = P / (avg_max_job_power * avg_num_servers + experiment_config.idle_power * avg_idle_servers)
    R_normalized = (2 * R) / (avg_max_job_power * experiment_config.server_count -
                              experiment_config.idle_power * experiment_config.server_count)

    return P_normalized, R_normalized
