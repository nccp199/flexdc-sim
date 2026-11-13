import argparse
import numpy as np
import pandas as pd
from peacsim.simulator import Simulator
from peacsim.run_simulator import init_job_table, init_node_table
from peacsim.parsing import experiment_config_reader as experiment_parser
from peacsim.parsing import cluster_profile_reader as cluster_parser
from peacsim.parsing import policy_config_reader as policy_parser
from peacsim.parsing import job_profile_reader as job_parser
from peacsim.AQA_runtime_policy import AQARuntimePolicy
from peacsim.parsing import gradient_descent_reader as gradient_parser
import datetime
from denormalize_PR import denormalize_PR
import os
import itertools
from calculate_qos_cost import calculate_qos_cost


def init_output(output_dir, experiment_config, policy_config):
    with open(output_dir + '/power_trace.csv', 'w') as f:
        f.write(
            'time,QoSemergency,numNeededForTracking,waitingSum,target,realSum,trackingError,QoSJobSum,standbyPower')
        for i in range(experiment_config.server_count):
            f.write(',client%d' % (i + 1))
        f.write(',estimateSum')
        for i in range(experiment_config.server_count):
            f.write(',state%d' % (i + 1))
        f.write('\n')
    with open(output_dir + '/PRtable.csv', 'w') as f:
        f.write('phase,P,R\n')
        f.write('%d,%d,%d\n' % (0, policy_config.P, policy_config.R))
    with open(output_dir + 'job_table.csv', 'w') as f:
        f.write('job_id,job_type_id,arrival_time,start_time,end_time,estimate_finished,update_time,realtime_qos,'
                'min_execution_time,qos_constraint,job_size\n')


def softplus(x):
    return np.log1p(np.exp(x))


def create_simulator_object(P_ratio, R_ratio, experiment_config, job_config, cluster_config, policy_config, output_dir):
    job_table = init_job_table(experiment_config, job_config)
    node_table = init_node_table(experiment_config)

    avg_max_job_power = sum(job_config.all_max_job_power.values()) / len(job_config.all_max_job_power)
    avg_num_servers = experiment_config.server_count * experiment_config.utilization
    avg_idle_servers = experiment_config.server_count - avg_num_servers
    P_ratio = round(P_ratio, 6)
    R_ratio = round(R_ratio, 6)
    policy_config._P_ratio = P_ratio
    policy_config._R_ratio = R_ratio
    policy_config._P = int(P_ratio * (avg_max_job_power * avg_num_servers +
                                      experiment_config.idle_power * avg_idle_servers))
    policy_config._R = int(R_ratio * (avg_max_job_power * experiment_config.server_count -
                                      (experiment_config.idle_power * experiment_config.server_count)) / 2)

    policy_mapping = {'AQA': (AQARuntimePolicy, (policy_config, experiment_config, job_config, node_table, job_table)),
                      'newAQA': ()}
    policy_class, constructor_args = policy_mapping[policy_config.runtime_policy_name]
    runtime_policy = policy_class(*constructor_args)

    simulator = Simulator(job_table, node_table, runtime_policy, experiment_config,
                          job_config, cluster_config, output_dir)
    return simulator


def objective_function(params, cost_function_config, policy_config, experiment_config, cluster_config, job_config, output_dir):
    Pbar, R, *weights = params

    base_weights_path = os.path.join(output_dir, 'base_weights.csv')
    pd.DataFrame({
        'job_type_id': np.array(list(job_config.all_jobs.values())),
        'weights': weights
    }).to_csv(base_weights_path, index=False)

    policy_config.policy_parameters['weights_path'] = base_weights_path
    simulator = create_simulator_object(Pbar, R, experiment_config, job_config, cluster_config,
                                        policy_config, output_dir)
    init_output(output_dir, experiment_config, policy_config)
    cost_power, cost_tracking = simulator.run()

    psi1, psi2 = cost_function_config.calculate_gradient_variables['psi1'], \
        cost_function_config.calculate_gradient_variables['psi2']

    tracking_cost_with_penalty = psi1 * cost_tracking * \
                                  (1 +  np.log(1 + np.exp(psi2 * (cost_tracking - 0.3))))
    cost_tracking = tracking_cost_with_penalty

    qos_constraints = list(simulator._job_config.all_job_qos_constraints.values())
    min_execution_times = list(simulator._job_config.all_min_execution_time.values())
    cost_qos = calculate_qos_cost(experiment_config=experiment_config, job_config=job_config, policy_config=policy_config,
                                  simulator=simulator, fit=True, fitdown=0.85, fitup=0.95, sim_hour=1,
                                  J=simulator._job_type_count, job_table=simulator.get_job_report(),
                                  QoS_constraint=qos_constraints, min_execution_time=min_execution_times,
                                  cost_function_config=cost_function_config
                                  )

    # Weighted sum for objective function
    objective_value = cost_power + cost_tracking + cost_qos
    print(f"Iteration Variables: Pbar={Pbar}, R={R}, Weights={weights}, "
          f"Cost={objective_value},"
          f"Power Cost={cost_power}, Tracking Cost={cost_tracking}, QoS Cost={cost_qos}")

    results_file = os.path.join(output_dir, 'grid_search_results.csv')
    results = [Pbar, R] + list(weights) + [objective_value, cost_power, cost_tracking, cost_qos] + \
              [simulator.ratio_of_job_type_qos_violation, simulator.tracking_error_at_90]
    with open(results_file, 'a') as f:
        f.write(','.join(map(str, results)) + '\n')

    return objective_value


def generate_weight_grid(lower, upper, step, count):
    """
    Generate all combinations of `count` weights in the range [lower, upper] (with given step)
    such that their sum is exactly 1 (within a small tolerance).
    """
    values = np.arange(lower, upper + step / 2, step)
    weight_combinations = []
    for combo in itertools.product(values, repeat=count):
        if abs(sum(combo) - 1.0) < 1e-6:
            weight_combinations.append(combo)
    return weight_combinations


def grid_search_parameters(cost_function_config, policy_config, experiment_config, cluster_config, job_config, output_dir,
                           num_points=10, weight_step=0.02):
    """
    Performs grid search over Pbar, R and the weight simplex.

    - Pbar is varied from 0.9 to 1.1 (num_points values).
    - R is varied from 0.5 to 0.7 (num_points values).
    - For 4 job types, weights are chosen from the range [0.2, 0.3] in increments of weight_step,
      and only combinations that sum to 1 are evaluated.
    """
    best_cost = float('inf')
    best_params = None

    # Generate weight combinations for 4 job types
    weight_combinations = generate_weight_grid(0.2, 0.3, weight_step, job_config.job_type_count)
    # weight_combinations = [[0.25, 0.25, 0.25, 0.25]]
    # weight_combinations = [[0.27999999999999997, 0.26, 0.22, 0.24]]

    print(f"Generated {len(weight_combinations)} valid weight combinations.")

    # CSV file for logging iterations
    results_file = os.path.join(output_dir, 'grid_search_results.csv')
    with open(results_file, 'w') as f:
        headers = ['Pbar', 'R'] + [f'Weight_{i}' for i in range(job_config.job_type_count)] + \
                  ['Total_Cost', 'Power_Cost', 'Tracking_Cost', 'QoS_Cost', 'QoS_Violation_Ratio', 'Tracking_Error_90th']
        f.write(','.join(headers) + '\n')

    # Loop over Pbar and R values
    for Pbar in np.linspace(0.9, 1.1, num_points):
        for R in np.linspace(0.5, 0.8, num_points):
            # Enforce constraint: Pbar must be greater than R.
            if Pbar <= R:
                continue
            # Loop over all valid weight combinations
            for weights in weight_combinations:
                params = [Pbar, R] + list(weights)
                cost = objective_function(params, cost_function_config, policy_config,
                                          experiment_config, cluster_config, job_config,
                                          output_dir)
                if cost < best_cost:
                    best_cost = cost
                    best_params = params
    return best_params


if __name__ == "__main__":

    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="Parse configuration file paths.")

    # Configuration files under configs folder
    parser.add_argument('--gradient-config', type=str, required=True,
                        help="Path to the gradient descent configuration file (e.g., gradient_descent.ini)")
    parser.add_argument('--experiment-config', type=str, required=True,
                        help="Path to the experiment configuration file (e.g., exp.ini)")
    parser.add_argument('--cluster-config', type=str, required=True,
                        help="Path to the cluster configuration file (e.g., cluster.ini)")
    parser.add_argument('--policy-name', type=str, required=True,
                        choices=['AQA'],
                        help="Name of the runtime policy. Options: AQA")
    parser.add_argument('--job-config', type=str, required=True,
                        help="Path to the workload mix having job power performance configuration file (e.g., workload/W10.ini)")

    # Additional parameters
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Custom output directory name for simulation results under src.peacsim.output.")
    parser.add_argument('--logger', type=str, required=False, default='', help="Available loggers: ['wandb']")

    args = parser.parse_args()
    policy_config = policy_parser.PolicyConfigReader.from_parameters(
        policy_name=args.policy_name,
        P=-1,
        R=-1,
        P_ratio=-1,
        R_ratio=-1,
        policy_parameters={}
    )

    # The gradient config is no longer used in grid search but is kept for compatibility.
    cost_function_config = gradient_parser.GradientDescentReader(args.gradient_config)

    experiment_config = experiment_parser.ExperimentConfigReader(args.experiment_config)
    cluster_config = cluster_parser.ClusterProfileReader(args.cluster_config)
    job_config = job_parser.JobProfileReader(args.job_config)

    output_dir = 'output/optimization/' + args.output_dir + '_' + start_time.strftime("%Y%m%d%H%M%S") + '/'
    os.makedirs(output_dir, exist_ok=True)
    print("saving results in", output_dir)

    optimal_params = grid_search_parameters(cost_function_config, policy_config, experiment_config,
                                            cluster_config, job_config, output_dir, num_points=10
                                            )
    print("Optimal Parameters:", optimal_params)
