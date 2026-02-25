import argparse
import numpy as np
import pandas as pd
import os
import datetime
import json
from distutils.util import strtobool
from peacsim.simulator import Simulator
from peacsim.run_simulator import init_job_table, init_node_table
from peacsim.parsing import experiment_config_reader as experiment_parser
from peacsim.parsing import cluster_profile_reader as cluster_parser
from peacsim.parsing import policy_config_reader as policy_parser
from peacsim.parsing import job_profile_reader as job_parser
from peacsim.AQA_runtime_policy import AQARuntimePolicy
from peacsim.parsing import simulated_annealing_reader as sa_config_parser
from calculate_qos_cost import calculate_qos_cost_sa, calculate_qos_cost_availability
from calculate_monetary_cost import calculate_monetary_cost
from denormalize_PR import denormalize_p_and_r_to_watts
import random
import wandb

# -------------------------------
# Helper: Create simulation object
# -------------------------------

job_table_ = 0
node_table_ = 0


def create_simulator_object(P_ratio, R_ratio, experiment_config, job_config, cluster_config, policy_config,
                            qos_constraint, output_dir):
    job_table = job_table_.copy()
    node_table = node_table_.copy()

    policy_config._P_ratio = P_ratio
    policy_config._R_ratio = R_ratio

    policy_config._P, policy_config._R = denormalize_p_and_r_to_watts(experiment_config, job_config,
                                                                      policy_config._P_ratio, policy_config._R_ratio)

    policy_mapping = {
        'AQA': (
            AQARuntimePolicy, (policy_config, experiment_config, job_config, node_table, job_table)),
        'flex-resource': (
            AQARuntimePolicy, (policy_config, experiment_config, job_config, node_table, job_table)),
        'priority-cap': (AQARuntimePolicy, (policy_config, experiment_config, job_config, node_table, job_table)),
        'proportional-cap': (AQARuntimePolicy, (policy_config, experiment_config, job_config, node_table, job_table))
        }

    policy_class, constructor_args = policy_mapping[policy_config.runtime_policy_name]
    runtime_policy = policy_class(*constructor_args)

    qos_violation_percentile_cut_off = 1 - qos_constraint
    simulator = Simulator(job_table, node_table, runtime_policy, experiment_config,
                          job_config, cluster_config, qos_violation_percentile_cut_off, output_dir)
    return simulator


# -------------------------------
# Helper: Initialize output files
# -------------------------------
def init_output(output_dir, experiment_config, policy_config):
    with open(os.path.join(output_dir, 'power_trace.csv'), 'w') as f:
        f.write('time,QoSemergency,numNeededForTracking,waitingSum,target,realSum,trackingError,QoSJobSum,standbyPower')
        for i in range(experiment_config.server_count):
            f.write(',client%d' % (i + 1))
        f.write(',estimateSum')
        for i in range(experiment_config.server_count):
            f.write(',state%d' % (i + 1))
        f.write('\n')
    with open(os.path.join(output_dir, 'PRtable.csv'), 'w') as f:
        f.write('phase,P,R\n')
        f.write('%d,%d,%d\n' % (0, policy_config.P, policy_config.R))
    with open(os.path.join(output_dir, 'job_table.csv'), 'w') as f:
        f.write('job_id,job_type_id,arrival_time,start_time,end_time,estimate_finished,update_time,realtime_qos,'
                'min_execution_time,qos_constraint,job_size\n')


def sa_evaluate(params, simulated_annealing_config, policy_config, experiment_config,
                cluster_config, job_config, output_dir, iteration):
    """
    Evaluation function for SA
    Runs a simulation with the given parameters and returns:
      - total_cost,
      - violation: simulator.ratio_of_job_type_qos_violation (should be 0),
      - tracking: simulator.tracking_error.
    """
    Pbar, R, *weights = params
    base_weights_path = os.path.join(output_dir, 'base_weights.csv')
    pd.DataFrame({
        'job_type_id': np.array(list(job_config.all_jobs.values())),
        'weights': weights
    }).to_csv(base_weights_path, index=False)

    policy_config.policy_parameters['weights_path'] = base_weights_path

    experiment_config._random_seed = iteration
    simulator = create_simulator_object(Pbar, R, experiment_config, job_config, cluster_config, policy_config,
                                        simulated_annealing_config.qos_constraint,
                                        output_dir)
    init_output(output_dir, experiment_config, policy_config)

    #  RUN SIMULATION  #
    cost_power_dummy, cost_tracking_raw = simulator.run()
    #  ==============  #

    psi1 = simulated_annealing_config.psi1
    psi2 = simulated_annealing_config.psi2

    cost_power = calculate_monetary_cost(policy_config.P, policy_config.R, simulated_annealing_config.program_type,
                                         sim_hour=experiment_config.simulation_duration / 3600,
                                         optimization_config=simulated_annealing_config)
    cost_tracking = psi1 * cost_tracking_raw * (1 + np.log(1 + np.exp(psi2 * (cost_tracking_raw - 0.3))))

    qos_constraints = list(simulator._job_config.all_job_qos_constraints.values())
    min_execution_times = list(simulator._job_config.all_min_execution_time.values())

    cost_qos, qos_costs_each = 0, []

    if policy_config.runtime_policy_name == 'flex-resource':
        cost_qos, qos_costs_each = calculate_qos_cost_availability(experiment_config.server_count, weights,
                                                                   policy_config.policy_parameters['probabilities'],
                                                                   output_dir,
                                                                   simulated_annealing_config)
    elif (policy_config.runtime_policy_name == 'AQA' or
          policy_config.runtime_policy_name == 'priority-cap' or
          policy_config.runtime_policy_name == 'proportional-cap'):
        cost_qos, qos_costs_each = calculate_qos_cost_sa(sim_hour=experiment_config.simulation_duration / 3600,
                                                         J=simulator._job_type_count,
                                                         job_table=simulator._job_table,
                                                         QoS_constraint=qos_constraints,
                                                         min_execution_time=min_execution_times,
                                                         cost_function_config=simulated_annealing_config,
                                                         get_each_term=True)

    objective_value = cost_power + cost_tracking + cost_qos
    total_cost = objective_value
    qos_violation_ratio = simulator.ratio_of_job_type_qos_violation
    tracking = simulator.tracking_error_at_90
    return total_cost, cost_power, cost_qos, cost_tracking, qos_violation_ratio, tracking, qos_costs_each


def smart_weight_update(params, policy_config, step_size=0.01, qos_cost_each=None):
    """
    Helpers for SA: Perturbation and bounds enforcement
    Update weights based on QoS degradation severity.

    - params: List containing [P, R, w1, w2, ..., wn]
    - step_size: Base step size for weight updates.
    - qos_cost_each: List of QoS degradation costs for each job type.
    """
    if qos_cost_each is None:
        raise ValueError("qos_cost_each must be provided!")

    new_params = params.copy()
    w = np.array(new_params[2:])  # Extract weights

    if policy_config.runtime_policy_name == 'flex-resource':
        w *= (1 + step_size * (2 * qos_cost_each - 1))  # Shift range to [-1,1]
    elif policy_config.runtime_policy_name == 'AQA':
        # Normalize qos_cost_each so the worst degradation is scaled to 1
        max_qos = max(qos_cost_each)
        if max_qos > 0:
            qos_cost_scaled = np.array(qos_cost_each) / max_qos  # Scale between 0 and 1
        else:
            qos_cost_scaled = np.zeros_like(qos_cost_each)  # No degradation, no updates

        # Adaptive weight update: Higher QoS degradation means Larger update
        w *= (1 + step_size * qos_cost_scaled)

    # Normalize weights to maintain sum constraint
    w = np.clip(w, 1e-6, None)  # Prevent zero weights
    w /= np.sum(w)  # Ensure weights sum to 1

    new_params[2:] = w.tolist()
    return new_params


def perturb(params, step_size_p, step_size_r, step_size_w, simulated_annealing_config, policy_config, qos_cost_each=None):
    """Perturb parameters with Gaussian noise; renormalize weights afterward."""
    new_params = params.copy()
    # Perturb Pbar and R
    new_params[0] += np.random.normal(0, 1 * step_size_p)

    if simulated_annealing_config.program_type == 'RSR':    # There is no R update on EDR
        new_params[1] += np.random.normal(0, 1 * step_size_r)

    if policy_config.runtime_policy_name == 'priority-cap':
        return new_params

    # Perturb weights and then renormalize
    if simulated_annealing_config.smart_weight:
        new_params = smart_weight_update(new_params, policy_config, step_size_w, qos_cost_each)
    else:
        w = np.array(new_params[2:])
        w += np.random.normal(0, step_size_w, size=w.shape)
        w = np.clip(w, 1e-6, None)
        w = w / np.sum(w)
        new_params[2:] = w.tolist()

    return new_params


def enforce_bounds(params, bounds, simulated_annealing_config):
    """Enforce parameter bounds and renormalize weights."""
    new_params = params.copy()
    # Bounds for Pbar and R
    new_params[0] = np.clip(new_params[0], bounds[0][0], bounds[0][1])
    if simulated_annealing_config.program_type == 'RSR':
        new_params[1] = np.clip(new_params[1], bounds[1][0], bounds[1][1])
    # For weights: clip and renormalize
    w = np.array(new_params[2:])
    w = np.clip(w, bounds[2][0], bounds[2][1])
    w = w / np.sum(w)
    new_params[2:] = w.tolist()
    return new_params


def simulated_annealing_optimize(simulated_annealing_config, policy_config, experiment_config,
                                 cluster_config, job_config, output_dir):

    # Define bounds: for Pbar, R, and each weight (assuming job_config.job_type_count weights)

    bounds = [(simulated_annealing_config.p_low, simulated_annealing_config.p_high),
              (simulated_annealing_config.r_low, simulated_annealing_config.r_high)] \
             + [(simulated_annealing_config.w_low, simulated_annealing_config.w_high)] * job_config.job_type_count

    # Initial candidate: use initial parameters from gradient descent config and equal weights
    p_init, r_init = simulated_annealing_config.p_init, simulated_annealing_config.r_init
    weights = [1 / job_config.job_type_count] * job_config.job_type_count
    current = [p_init, r_init] + weights
    current = enforce_bounds(current, bounds, simulated_annealing_config)

    # Evaluate initial candidate
    current_cost, cost_power, cost_qos, cost_tracking, \
        current_qos_violation_ratio, current_tracking_violation, qos_cost_each \
        = sa_evaluate(current, simulated_annealing_config, policy_config,
                      experiment_config, cluster_config, job_config, output_dir, iteration=0)

    # Penalty: if QoS violation is nonzero or tracking error exceeds 0.3, add heavy penalty
    penalty_current = 0  # 10 * current_qos_violation_ratio + 10 * max(current_tracking_violation - 0.3, 0)
    current_total_cost = current_cost + penalty_current

    best = current.copy()
    best_total_cost = current_total_cost

    temperature = simulated_annealing_config.temperature

    step_size_p = simulated_annealing_config.p_step
    step_size_r = simulated_annealing_config.r_step
    step_size_w = simulated_annealing_config.w_step

    for i in range(simulated_annealing_config.max_iter):
        print("==============  Iteration", i, "==============")

        if i == 200:
            step_size_p /= 2
            step_size_r /= 2
            step_size_w /= 2
        if i == 300:
            step_size_p /= 2
            step_size_r /= 2
            step_size_w /= 2

        candidate = perturb(current,
                            step_size_p,
                            step_size_r,
                            step_size_w,
                            simulated_annealing_config,
                            policy_config,
                            qos_cost_each=qos_cost_each)

        candidate = enforce_bounds(candidate, bounds, simulated_annealing_config)
        cost, cost_power, cost_qos, cost_tracking, qos_violation_ratio, tracking_at_90, qos_cost_each = \
            sa_evaluate(candidate, simulated_annealing_config, policy_config,
                        experiment_config, cluster_config, job_config, output_dir, iteration=i+1)
        weights = {i: candidate[i + 2] for i in range(len(candidate) - 2)}

        tracking_penalty = 0  # np.exp(tracking_at_90-0.3) if tracking_at_90 > 0.3 else 0
        qos_penalty = 0  # np.exp(qos_violation_ratio) - 1
        penalty = 10 * qos_penalty + 30 * tracking_penalty

        candidate_total_cost = cost + penalty

        wandb.log({"P": candidate[0], "R": candidate[1], "Cost": candidate_total_cost,
                   "Cost Power": cost_power,
                   "Cost QoS": cost_qos,
                   "Cost Tracking": cost_tracking,
                   "QoS penalty": qos_penalty,
                   "Tracking penalty": tracking_penalty,
                   "weights": weights,
                   "ratio_of_job_type_qos_violation": qos_violation_ratio,
                   "tracking_error_at_90": tracking_at_90,
                   "temperature": temperature})

        print("Candidate Parameters:", candidate)

        if qos_violation_ratio == 0 and tracking_at_90 < 0.3:
            print("Found a feasible solution")
        delta = candidate_total_cost - current_total_cost

        # Accept candidate if cost is lower or probabilistically if worse
        if delta < 0 or np.random.rand() < np.exp(-delta / temperature):
            print("Candidate Parameters Accepted:", candidate)
            current = candidate
            current_total_cost = candidate_total_cost
            # Track the best found candidate
            if candidate_total_cost < best_total_cost:
                best = candidate, i
                best_total_cost = candidate_total_cost

        # Cool down the temperature
        temperature *= simulated_annealing_config.cooling_rate
        print(
            f"Iteration {i}: Current Total Cost = {current_total_cost:.4f}, Best Total Cost = {best_total_cost:.4f}")

    return best


# -------------------------------
# Main block for running SA (if run as script)
# -------------------------------
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
                        choices=['AQA', 'flex-resource', 'priority-cap', 'proportional-cap'],
                        help="Name of the runtime policy. Options: AQA, flex-resource, priority-cap, proportional-cap")
    parser.add_argument('--job-config', type=str, required=True,
                        help="Path to the workload mix having "
                             "job power performance configuration file (e.g., workload/W10.ini)")

    # Additional parameters
    parser.add_argument('--output-dir', type=str, required=True,
                        help="custom output directory name for simulation results under src.peacsim.output.")

    parser.add_argument('--logger', type=str, required=False, default='', help="available loggers: ['wandb']")

    parser.add_argument('--probabilities', type=str, required=False, default=None,
                        help="probabilities of resource allocation (comma separated for each job type)")
    parser.add_argument('--power-cap-percentage', type=float, required=False, default=1.0)

    parser.add_argument('--node-state-control',
                        type=lambda x: bool(strtobool(x)),
                        default=True,
                        help="whether to control node state (e.g., idle/active) during simulation (true/false)"
                             " (default: True)")

    args = parser.parse_args()

    policy_parameters = {}
    if args.probabilities is not None:
        probabilities = [float(p) for p in args.probabilities.split(',')]
        policy_parameters['probabilities'] = probabilities
    if args.power_cap_percentage is not None:
        policy_parameters['power_cap_percentage'] = args.power_cap_percentage
    if args.node_state_control is not None:
        policy_parameters['node_state_control'] = args.node_state_control

    policy_config = policy_parser.PolicyConfigReader.from_parameters(
        policy_name=args.policy_name,
        P=-1,
        R=-1,
        P_ratio=-1,
        R_ratio=-1,
        policy_parameters=policy_parameters
    )

    simulated_annealing_config = sa_config_parser.SimulatedAnnealingConfigReader(args.gradient_config)
    experiment_config = experiment_parser.ExperimentConfigReader(args.experiment_config)
    cluster_config = cluster_parser.ClusterProfileReader(args.cluster_config)
    job_config = job_parser.JobProfileReader(args.job_config)
    output_dir_timestamped = args.output_dir + '_' + start_time.strftime("%Y%m%d%H%M%S") + '/'
    output_dir = 'output/optimization/' + output_dir_timestamped
    os.makedirs(output_dir, exist_ok=True)
    print("Output Directory: ", output_dir)
    print("Saving optimization configurations in", output_dir + "configurations.json")

    configurations = {
        "gradient_config": simulated_annealing_config.to_dict(),
        "experiment_config": experiment_config.to_dict(),
        "cluster_config": cluster_config.to_dict(),
        "policy_config": policy_config.to_dict(),
        "job_config": job_config.to_dict()
    }

    with open(output_dir + "configurations.json", "w") as file:
        json.dump(configurations, file, indent=4)

    np.random.seed(experiment_config.random_seed)
    random.seed(experiment_config.random_seed)

    wandb_configs = {**simulated_annealing_config.to_dict(),
                     **experiment_config.to_dict(),
                     'policy_name': policy_config.runtime_policy_name,
                     'experiment_config': experiment_config.to_dict(),
                     'workload_mix': job_config.workload_mix_name.split('/')[-1],
                     'beta': simulated_annealing_config.beta,
                     'rho': simulated_annealing_config.rho,
                     'temperature': simulated_annealing_config.temperature,
                     'cooling_rate': simulated_annealing_config.cooling_rate,
                     'smart_weight': simulated_annealing_config.smart_weight,
                     'power_cap_percentage': args.power_cap_percentage,
                     'node_state_control': args.node_state_control,
                     }

    wandb.init(project='flexdc_experiments', config=wandb_configs, name=output_dir_timestamped)

    job_table_ = init_job_table(experiment_config, job_config)
    node_table_ = init_node_table(experiment_config)

    best_params, iteration_index = simulated_annealing_optimize(simulated_annealing_config, policy_config, experiment_config,
                                                                cluster_config, job_config, output_dir)

    print("Optimal Parameters (via SA):", best_params, "found at iteration:", iteration_index)

    p_normalized, r_normalized = best_params[0], best_params[1]

    opt_weights = pd.DataFrame({
        'job_type_id': np.array(list(job_config.all_jobs.values())),
        'weights': best_params[2:]
    })

    opt_weights_path = output_dir + "/opt_weights.csv"
    opt_weights.to_csv(opt_weights_path, index=False)
    policy_parameters['weights_path'] = opt_weights_path
    P, R = denormalize_p_and_r_to_watts(experiment_config, job_config, p_normalized, r_normalized)

    optimal_policy_config = policy_parser.PolicyConfigReader.from_parameters(
        policy_name=args.policy_name,
        P=P,
        R=R,
        P_ratio=p_normalized,
        R_ratio=r_normalized,
        policy_parameters=policy_parameters
    )
    print("Saving optimal_policy.ini in the output directory:", output_dir)
    optimal_policy_config.write_policy_ini(policy_name=args.policy_name, policy_parameters=policy_parameters,
                                           file_path=output_dir + "/optimal_policy.ini")
    experiment_config.write_experiment_config(output_dir + "experiment_config.ini")
