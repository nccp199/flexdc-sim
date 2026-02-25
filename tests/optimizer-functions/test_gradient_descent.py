import unittest
from unittest.mock import MagicMock
import numpy as np
from peacsim.parsing import experiment_config_reader as experiment_parser
from peacsim.parsing import cluster_profile_reader as cluster_parser
from peacsim.parsing import policy_config_reader as policy_parser
from peacsim.parsing import job_profile_reader as job_parser
from peacsim.parsing import gradient_descent_reader as gradient_parser
from src.peacsim.gradient_descent import GradientDescentOptimizer
from src.peacsim.simulator import Simulator
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR)


class TestGradientDescentOptimizer(unittest.TestCase):
    def setUp(self):
        self.gradient_config = gradient_parser.GradientDescentReader(os.path.join(PROJECT_ROOT, "../test-configs", "test-optimizer", "test_gradient_descent.ini"))
        self.logger = None

        self.experiment_config = MagicMock()
        self.cluster_config = MagicMock()
        self.policy_config = MagicMock()
        self.job_config = MagicMock()
        self.output_dir = "test_output_dir"
        self.logger = None

        #self.optimizer = GradientDescentOptimizer( self.gradient_config, self.experiment_config, self.cluster_config, self.policy_config, self.job_config, self.output_dir, self.logger)

        # Mock the simulator
        self.simulator = MagicMock(spec=Simulator)
        self.simulator._server_count = 10
        self.simulator._runtime_policy = MagicMock()
        self.simulator._runtime_policy._P = 100
        self.simulator._runtime_policy._R = 50
        self.simulator._runtime_policy.weighted_average_power = 75
        self.simulator._runtime_policy.weights = {0: 0.5, 1: 0.5}
        self.simulator._idle_power = 10
        self.simulator._utilization = 0.8
        self.simulator._simulation_duration = 7200  # 2 hours

        # Mock job config values
        self.job_config.all_job_qos_constraints = {0: 1.0, 1: 1.0}
        self.job_config.all_min_execution_time = {0: 3600, 1: 3600}
        self.job_config.all_max_job_power = {0: 50, 1: 50}
        self.job_config.all_job_size = {0: 1, 1: 1}
        self.job_config.job_type_count = 2
        self.job_config.all_jobs = {0: 0, 1: 1}

    def test_calculate_gradient(self):
        iteration = 0
        beta = 0.1
        rho = 0.1
        sim_hour = 2
        alpha = {0: 0.5, 1: 0.5}
        delay_prob = [0.1, 0.1]
        job_power_max = [50, 50]
        job_size = [1, 1]

        #dCdP, dCdR, dCdw, cost_power, cost_error, cost_qos, cost_standby = self.optimizer.calculate_gradient(
        #    iteration, beta, rho, sim_hour, self.simulator, alpha, delay_prob, job_power_max, job_size
        #)

        # Assertions to check if the outputs are as expected
        #self.assertIsInstance(dCdP, float)
        #self.assertIsInstance(dCdR, float)
        #self.assertIsInstance(dCdw, list)
        #self.assertIsInstance(cost_power, float)
        #self.assertIsInstance(cost_error, float)
        #self.assertIsInstance(cost_qos, float)
        #self.assertIsInstance(cost_standby, float)


if __name__ == '__main__':
    unittest.main()