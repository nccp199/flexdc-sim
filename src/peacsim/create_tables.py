import itertools
import numpy as np
import random


def init_job_table_adaptive(experiment_config, job_config, trials=5):
    """Generate job tables over multiple trials and select the median trial to smooth randomness."""

    def arrival_generator(r):
        """Generates job arrival times using an exponential distribution (Poisson process)."""
        t = 0
        while True:
            t += random.expovariate(r)
            yield t

    num_job_types = len(job_config.all_jobs)
    server_count = experiment_config.server_count
    utilization = experiment_config.utilization
    simulation_duration = experiment_config.simulation_duration
    job_sizes = list(job_config.all_job_size.values())  # Convert to list for iteration

    arrival_rates = [
        utilization * server_count / float(num_job_types) /
        job_config.all_min_execution_time[i] / job_config.all_job_size[i]
        for i in range(num_job_types)
    ]

    trial_job_tables = []

    for _ in range(trials):
        initial_job_type_ids = []
        initial_arrival_times = []
        arrivals = []

        # Keep Initial Job Prefill Step
        for i in range(0, int(server_count / (2 * sum(job_sizes)) * (2 * len(job_sizes)))):
            initial_job_type_ids.append(random.randint(0, num_job_types - 1))
        initial_arrival_times = [0] * len(initial_job_type_ids)

        for i, rate in enumerate(arrival_rates):
            generator = arrival_generator(rate)
            arrivals += [(arrival, i) for arrival in
                         itertools.takewhile(lambda time: time < simulation_duration, generator)]

        sorted_arrivals = sorted(arrivals, key=lambda x: x[0])

        for s in sorted_arrivals:
            initial_job_type_ids.append(s[1])
            initial_arrival_times.append(round(s[0]))

        job_table_size = len(initial_job_type_ids)
        trial_job_tables.append((job_table_size, initial_arrival_times, initial_job_type_ids))

    # Select the Median Trial Instead of the Closest to the Average
    trial_job_tables.sort(key=lambda x: x[0])  # Sort by job table size
    median_trial = trial_job_tables[len(trial_job_tables) // 2]  # Select median job count trial

    job_table_size, final_arrival_times, final_job_type_ids = median_trial

    job_id = np.arange(job_table_size, dtype=np.float64)  # 0
    job_type_id = final_job_type_ids  # 1
    arrival_time = final_arrival_times  # 2
    start_time = np.full(job_table_size, -1, dtype=np.float64)  # 3
    end_time = np.full(job_table_size, -1, dtype=np.float64)  # 4
    estimate_finished = np.zeros(job_table_size, dtype=np.float64)  # 5
    update_time = final_arrival_times  # 6
    realtime_qos = np.zeros(job_table_size, dtype=np.float64)  # 7
    min_execution_time = np.array(
        [job_config.all_min_execution_time[job_type] for job_type in final_job_type_ids],
        dtype=np.float64)  # 8
    qos_constraint = np.array(
        [job_config.all_job_qos_constraints[job_type] for job_type in final_job_type_ids],
        dtype=np.float64)  # 9

    return np.array([job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time,
                     realtime_qos, min_execution_time, qos_constraint])

def generate_arrival_rates(experiment_config, job_config):
    """
    Calculate the arrival rates for each job type based on the given configuration.
    """
    num_job_types = len(job_config.all_jobs)
    server_count = experiment_config.server_count
    utilization = experiment_config.utilization
    return [
        utilization * server_count / float(num_job_types) /
        job_config.all_min_execution_time[i] / job_config.all_job_size[i]
        for i in range(num_job_types)
    ]


def generate_arrivals(arrival_rates, simulation_duration, arrival_generator, spike_rates=None, spike_start=None, spike_end=None):
    """
    Generate arrivals for each job type and return a sorted list of arrival times and job type IDs.

    :param arrival_rates: List of normal arrival rates.
    :param simulation_duration: Total simulation time.
    :param arrival_generator: Function to generate arrivals using a rate.
    :param spike_rates: Optional list of spike arrival rates for workload spikes.
    :param spike_start: Start time (in minutes) for workload spike.
    :param spike_end: End time (in minutes) for workload spike.
    """
    arrivals = []
    for i, rate in enumerate(arrival_rates):
        if spike_rates and spike_start is not None and spike_end is not None:
            generator = arrival_generator(rate, spike_r=spike_rates[i], spike_start=spike_start, spike_end=spike_end)
        else:
            generator = arrival_generator(rate)
        arrivals += [(arrival, i) for arrival in itertools.takewhile(lambda time: time < simulation_duration, generator)]

    return sorted(arrivals, key=lambda x: x[0])


def generate_job_table(job_table_size, initial_job_type_ids, initial_arrival_times, job_config):
    """
    Creates and returns a NumPy array containing the job table with various job properties.
    """
    job_id = np.arange(job_table_size, dtype=np.float64)
    arrival_time = initial_arrival_times
    start_time = np.full(job_table_size, -1, dtype=np.float64)
    end_time = np.full(job_table_size, -1, dtype=np.float64)
    estimate_finished = np.zeros(job_table_size, dtype=np.float64)
    update_time = arrival_time
    realtime_qos = np.zeros(job_table_size, dtype=np.float64)
    min_execution_time = np.array(
        [job_config.all_min_execution_time[job_type] for job_type in initial_job_type_ids],
        dtype=np.float64
    )
    qos_constraint = np.array(
        [job_config.all_job_qos_constraints[job_type] for job_type in initial_job_type_ids],
        dtype=np.float64
    )

    return np.array([
        job_id, initial_job_type_ids, arrival_time, start_time, end_time, estimate_finished, update_time,
        realtime_qos, min_execution_time, qos_constraint
    ])


def arrival_generator(r, spike_r=None, spike_start=None, spike_end=None):
    """
    Generates job arrival times using an exponential distribution,
    with an optional spike in arrival rates during a specified time range.
    """
    t = 0
    while True:
        # Choose the appropriate rate based on current time
        current_rate = spike_r if (spike_r and spike_start * 60 <= t < spike_end * 60) else r
        t += random.expovariate(current_rate)
        yield t


def init_job_table(experiment_config, job_config):
    arrival_type = experiment_config.workload_trace
    if arrival_type == 'poisson':
        return init_job_table_poisson(experiment_config, job_config)
    elif arrival_type == 'spike':
        return init_job_table_with_spike(experiment_config, job_config)
    else:
        raise ValueError(f'Unknown arrival type: {arrival_type}')

def init_job_table_poisson(experiment_config, job_config):
    """
    Standard initialization of the job table without workload spikes.
    """
    arrival_rates = generate_arrival_rates(experiment_config, job_config)
    simulation_duration = experiment_config.simulation_duration

    # Generate job prefill
    initial_job_type_ids = []
    job_sizes = job_config.all_job_size.values()
    for _ in range(0, int(experiment_config.server_count / (2 * sum(job_sizes)) * (2 * len(job_sizes)))):
        initial_job_type_ids.append(random.randint(0, len(job_config.all_jobs) - 1))
    initial_arrival_times = [0] * len(initial_job_type_ids)

    # Generate arrivals
    sorted_arrivals = generate_arrivals(arrival_rates, simulation_duration, arrival_generator)

    for arrival, job_type in sorted_arrivals:
        initial_job_type_ids.append(job_type)
        initial_arrival_times.append(round(arrival))

    # Create job table
    return generate_job_table(len(initial_job_type_ids), initial_job_type_ids, initial_arrival_times, job_config)


def init_job_table_with_spike(experiment_config, job_config):
    """
    Initialization of the job table with a workload spike for 10 minutes between the 30th and 40th minutes.
    """
    arrival_rates = generate_arrival_rates(experiment_config, job_config)
    spike_rates = [rate * experiment_config.spike_rate for rate in arrival_rates]
    print("SPIKE", experiment_config.spike_rate)
    simulation_duration = experiment_config.simulation_duration

    # Generate job prefill
    initial_job_type_ids = []
    job_sizes = job_config.all_job_size.values()
    for _ in range(0, int(experiment_config.server_count / (2 * sum(job_sizes)) * (2 * len(job_sizes)))):
        initial_job_type_ids.append(random.randint(0, len(job_config.all_jobs) - 1))
    initial_arrival_times = [0] * len(initial_job_type_ids)

    # Generate arrivals with spike
    sorted_arrivals = generate_arrivals(
        arrival_rates, simulation_duration, arrival_generator,
        spike_rates=spike_rates, spike_start=20, spike_end=40
    )

    for arrival, job_type in sorted_arrivals:
        initial_job_type_ids.append(job_type)
        initial_arrival_times.append(round(arrival))

    # Create job table
    return generate_job_table(len(initial_job_type_ids), initial_job_type_ids, initial_arrival_times, job_config)

def init_node_table(experiment_config):
    server_count = int(experiment_config.server_count)

    node_id = np.arange(server_count)  # 0
    server_job_id = np.full(server_count, -1)  # 1
    server_state = np.full(server_count, -1)  # 3
    server_power = np.full(server_count, experiment_config.idle_power)  # 3
    estimate_power = np.full(server_count, experiment_config.idle_power)  # 4
    job_progress = np.zeros(server_count, dtype=np.float64)  # 5
    min_exec_time = np.zeros(server_count)  # 6
    max_exec_time = np.zeros(server_count)  # 7
    min_job_power = np.full(server_count, experiment_config.idle_power)  # 8
    max_job_power = np.full(server_count, experiment_config.idle_power)  # 9
    just_started = np.full(server_count, -1) # 10
    control_power = np.full(server_count, experiment_config.idle_power) # 11
    return np.array([node_id, server_job_id, server_state, server_power, estimate_power, job_progress,
                     min_exec_time, max_exec_time, min_job_power, max_job_power, just_started, control_power])
