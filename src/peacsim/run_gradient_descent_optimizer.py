from peacsim.parsing import experiment_config_reader as experiment_parser
from peacsim.parsing import cluster_profile_reader as cluster_parser
from peacsim.parsing import policy_config_reader as policy_parser
from peacsim.parsing import job_profile_reader as job_parser
from peacsim.parsing import gradient_descent_reader as gradient_parser
from peacsim.gradient_descent import GradientDescentOptimizer
from peacsim.gradient_descent_plot import plot_gradient_descent
import argparse
import datetime


def main():
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
                        help="Path to the workload mix having "
                             "job power performance configuration file (e.g., workload/W10.ini)")

    # Additional parameters
    parser.add_argument('--output-dir', type=str, required=True,
                        help="custom output directory name for simulation results under src.peacsim.output.")

    parser.add_argument('--logger', type=str, required=False, default='', help="available loggers: ['wandb']")

    args = parser.parse_args()

    policy_config = policy_parser.PolicyConfigReader.from_parameters(
        policy_name=args.policy_name,
        P=-1,
        R=-1,
        P_ratio=-1,
        R_ratio=-1,
        policy_parameters={"node_count_control": True}
    )

    gradient_config = gradient_parser.GradientDescentReader(
        args.gradient_config)  # '../../configs/gradient_descent/gradient_descent_EDR.ini'

    experiment_config = experiment_parser.ExperimentConfigReader(
        args.experiment_config)  # '../../configs/experiment/exp.ini'
    cluster_config = cluster_parser.ClusterProfileReader(args.cluster_config)
    job_config = job_parser.JobProfileReader(args.job_config)

    output_dir = 'output/optimization/' + args.output_dir + '_' + start_time.strftime("%Y%m%d%H%M%S") + '/'

    optimizer = GradientDescentOptimizer(gradient_config,
                                         experiment_config,
                                         cluster_config,
                                         policy_config,
                                         job_config,
                                         output_dir,
                                         args.logger)
    optimizer.run()
    print('Optimizer finished in: ', (datetime.datetime.now() - start_time))

    print('Plotting optimization results at ' + output_dir)
    plot_gradient_descent(output_dir + "/GD.csv")


if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.set_start_method('spawn')
    main()
