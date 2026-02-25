import datetime

from peacsim.parsing import experiment_config_reader as experiment_parser
from peacsim.parsing import cluster_profile_reader as cluster_parser
from peacsim.parsing import policy_config_reader as policy_parser
from peacsim.parsing import job_profile_reader as job_parser
from peacsim.simulator import Simulator
from peacsim.flex_resource_policy import FlexResourcePolicy
from peacsim.aqa_runtimepolicy import AQARuntimePolicy
from peacsim.proportional_cap_policy import ProportionalCapPolicy
from peacsim.priority_cap_policy import PriorityCapPolicy
from peacsim.denormalize_PR import denormalize_p_and_r_to_watts
from peacsim.create_tables import init_job_table, init_node_table
import random
import numpy as np
import argparse


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


def main():
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="Parse configuration file paths.")

    parser.add_argument('--experiment-config', type=str, required=True,
                        help="Path to the experiment configuration file (e.g., exp.ini)")
    parser.add_argument('--cluster-config', type=str, required=True,
                        help="Path to the cluster configuration file (e.g., cluster.ini)")
    parser.add_argument('--policy-config', type=str, required=True,
                        help="Path to the runtime policy configuration file (e.g., AQA.ini)")
    parser.add_argument('--job-config', type=str, required=True,
                        help="Path to the workload mix having "
                             "job power performance configuration file (e.g., workload/W10.ini)")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="custom output directory name for simulation results under src.peacsim.output.")
    parser.add_argument('--convert-from-normalized-PR', action='store_true', default=False,
                        help="Use normalized P and R values and convert to Watts to run the simulation.")
    parser.add_argument('--plot-results', action='store_true', default=False,
                        help="Plotting for power tracking and QoS.")
    parser.add_argument('--qos-percentile', default=0.9, type=float,
                        help="qos requirement percentile cut-off for QoS violation.")

    args = parser.parse_args()

    cluster_config = cluster_parser.ClusterProfileReader(args.cluster_config)  # '../../configs/cluster/cluster.ini'
    policy_config = policy_parser.PolicyConfigReader(args.policy_config)  # '../../configs/policy/AQA-W10.ini'
    experiment_config = experiment_parser.ExperimentConfigReader(args.experiment_config)  # '../../configs/experiment/exp.ini'
    job_config = job_parser.JobProfileReader(args.job_config)  # '../../configs/workload/W10.ini'

    # Set random seed
    np.random.seed(experiment_config.random_seed)
    random.seed(experiment_config.random_seed)

    job_table = init_job_table(experiment_config, job_config)
    node_table = init_node_table(experiment_config)

    if args.convert_from_normalized_PR:
        P, R = denormalize_p_and_r_to_watts(experiment_config, job_config, policy_config.P_ratio, policy_config.R_ratio)
        policy_config._P = P
        policy_config._R = R

    if policy_config.runtime_policy_name == 'flex-resource':
        policy_config.policy_parameters['probabilities'] = \
            [float(p) for p in policy_config.policy_parameters['probabilities'].split(',')]
    policy_mapping = {
        'AQA': (
            AQARuntimePolicy, (policy_config, experiment_config, job_config, node_table, job_table)),
        'flex-resource': (
            FlexResourcePolicy, (policy_config, experiment_config, job_config, node_table, job_table)),
        'priority-cap': (
            PriorityCapPolicy, (policy_config, experiment_config, job_config, node_table, job_table)),
        'proportional-cap': (
            ProportionalCapPolicy, (policy_config, experiment_config, job_config, node_table, job_table))
    }

    policy_class, constructor_args = policy_mapping[policy_config.runtime_policy_name]
    runtime_policy = policy_class(*constructor_args)

    output_dir = 'output/simulation/' + args.output_dir + '_' + start_time.strftime("%Y%m%d%H%M%S") + '/'

    simulator = Simulator(job_table,
                          node_table,
                          runtime_policy,
                          experiment_config,
                          job_config,
                          cluster_config,
                          qos_violation_percentile_cut_off=args.qos_percentile,
                          output_dir=output_dir)

    init_output(output_dir, experiment_config, policy_config)
    simulator.run()

    print('finished in %d seconds.' % (datetime.datetime.now() - start_time).seconds)

    experiment_config.write_experiment_config(output_dir + 'experiment_config.ini')
    job_config.write_job_profile_config(output_dir + 'job_config.ini')


if __name__ == "__main__":
    main()
