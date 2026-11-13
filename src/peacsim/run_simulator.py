import datetime
from peacsim import simulation_output

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
from peacsim.qos_evaluation import prepare_qos_evaluation
from peacsim.metrics.energy_cost import ElectricityTariff
import random
import numpy as np
import argparse


def init_output(output_dir, experiment_config, policy_config):
    """Compatibility entry point for the shared CSV initialization."""
    return simulation_output.init_output(output_dir, experiment_config, policy_config)


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
    parser.add_argument('--scheduling-seed', type=int, default=None,
                        help="Optional runtime seed for replaying an optimizer evaluation; "
                             "the experiment seed still generates the workload.")
    parser.add_argument('--it-price-per-kwh', type=float, default=None,
                        help='Optional fixed price for the IT-only electricity estimate.')
    parser.add_argument('--it-price-currency', default=None,
                        help='Currency label for --it-price-per-kwh, e.g. CNY or USD.')
    parser.add_argument('--it-price-source', default=None,
                        help='Source or explicit assumption behind the supplied IT electricity price.')

    args = parser.parse_args()
    it_tariff = None
    price_fields = (args.it_price_per_kwh, args.it_price_currency, args.it_price_source)
    if any(value is not None for value in price_fields):
        if any(value is None for value in price_fields):
            parser.error('Supply --it-price-per-kwh, --it-price-currency and --it-price-source together.')
        try:
            it_tariff = ElectricityTariff(*price_fields)
        except ValueError as exc:
            parser.error(str(exc))

    cluster_config = cluster_parser.ClusterProfileReader(args.cluster_config)  # '../../configs/cluster/cluster.ini'
    policy_config = policy_parser.PolicyConfigReader(args.policy_config)  # '../../configs/policy/AQA-W10.ini'
    experiment_config = experiment_parser.ExperimentConfigReader(args.experiment_config)  # '../../configs/experiment/exp.ini'
    job_config = job_parser.JobProfileReader(args.job_config)  # '../../configs/workload/W10.ini'
    prepare_qos_evaluation(experiment_config, job_config)

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
                          output_dir=output_dir, it_tariff=it_tariff)

    init_output(output_dir, experiment_config, policy_config)
    if args.scheduling_seed is not None:
        simulator._random_seed = args.scheduling_seed
    simulator.run()
    energy_cost = simulator.get_energy_cost_report()
    estimate = energy_cost['it_electricity_cost_estimate']
    print('Observed IT energy: %.6f kWh; electricity estimate: %s' % (
        energy_cost['it_energy_kwh'], 'price not configured' if estimate['amount'] is None else
        '%g %s (%s)' % (estimate['amount'], estimate['currency'], estimate['status'])))

    print('finished in %d seconds.' % (datetime.datetime.now() - start_time).seconds)

    experiment_config.write_experiment_config(output_dir + 'experiment_config.ini')
    job_config.write_job_profile_config(output_dir + 'job_config.ini')


if __name__ == "__main__":
    main()
