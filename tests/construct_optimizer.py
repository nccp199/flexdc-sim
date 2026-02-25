import os
import datetime
from peacsim.gradient_descent import GradientDescentOptimizer

from peacsim.parsing import (
    experiment_config_reader,
    policy_config_reader,
    job_profile_reader,
    cluster_profile_reader,
    gradient_descent_reader
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR)

EXP_FILENAME = os.path.join(PROJECT_ROOT, "test-configs", "test-simulator", "test-exp.ini")
JOB_FILENAME = os.path.join(PROJECT_ROOT, "test-configs", "test-simulator", "test-job.ini")
POLICY_FILENAME = os.path.join(PROJECT_ROOT, "test-configs", "test-simulator", "test-policy.ini")
GRADIENT_FILENAME = os.path.join(PROJECT_ROOT, "test-configs", "test-optimizer", "test_gradient_descent.ini")


class ClusterConfig:
    def __init__(self):
        self.dell_1_idle_watts = 91.9
        self.dell_2_idle_watts = 91.8
        self.dell_3_idle_watts = 90.0

    def to_dict(self):
        return {
            "dell_1_idle_watts": self.dell_1_idle_watts,
            "dell_2_idle_watts": self.dell_2_idle_watts,
            "dell_3_idle_watts": self.dell_3_idle_watts,
        }


def construct_optimizer(output_dir=None, logger=None):
    """
    Constructs a GradientDescentOptimizer using the test configuration files.

    Args:
        output_dir (str): Directory where optimizer output/configurations will be saved.
                          If None, a directory will be created under output/optimization/.
        logger (optional): Logger identifier (e.g., 'wandb'). Defaults to None.

    Returns:
        GradientDescentOptimizer: The constructed optimizer instance.
    """
    if output_dir is None:
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = os.path.join("output", "optimization", "test_optimizer_" + now)
    os.makedirs(output_dir, exist_ok=True)

    gradient_config = gradient_descent_reader.GradientDescentReader(GRADIENT_FILENAME)

    experiment_config = experiment_config_reader.ExperimentConfigReader(EXP_FILENAME)
    job_config = job_profile_reader.JobProfileReader(JOB_FILENAME)
    policy_config = policy_config_reader.PolicyConfigReader(POLICY_FILENAME)
    cluster_config = ClusterConfig()

    optimizer = GradientDescentOptimizer(
        gradient_config,
        experiment_config,
        cluster_config,
        policy_config,
        job_config,
        output_dir,
        logger
    )
    return optimizer
