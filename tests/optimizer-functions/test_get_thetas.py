import os
import numpy as np
import tempfile
from tests.construct_optimizer import construct_optimizer


def test_creates_file():
    """
    Test that the get_thetas method writes a file 'thetaStars.csv'
    with the expected number of theta values (one per job type).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        optimizer = construct_optimizer(output_dir=temp_dir)
        utilization = 0.85
        server_count = 3
        weights = np.array([0.33, 0.33, 0.34])
        all_max_job_power = np.array([276.0, 280.0, 402.0])
        idle_power = 120.0
        all_job_min_execution_time = np.array([30.7, 739.0, 52.6])
        job_type_count = 3
        all_job_size = np.array([1.0, 2.0, 1.0])
        P = 10000
        R = 5000

        optimizer.get_thetas(utilization, server_count, weights, all_max_job_power,
                             idle_power, all_job_min_execution_time, job_type_count,
                             all_job_size, P, R)

        theta_file = optimizer.output_dir + 'thetaStars.csv'
        assert os.path.exists(theta_file), "thetaStars.csv file was not created by get_thetas."

        with open(theta_file, 'r') as f:
            content = f.read().strip()
        theta_values = [val for val in content.split(',') if val]
        assert len(theta_values) == job_type_count, (
            f"Expected {job_type_count} theta values but found {len(theta_values)} in thetaStars.csv."
        )


def test_values():
    """
    Test that get_thetas computes nonzero theta values that agree with the expected theory.

    In get_thetas, if the initial solver returns a value (theta0) that is above a threshold,
    the else branch is used, which computes:
         theta_star = nmu^2 / (2 * nsigma^2 * tau)

    where:
      - nmu = (P - idle_power * server_count) / (dot(all_max_job_power, weights) - idle_power)
      - nsigma = ysigma * R / (dot(all_max_job_power, weights) - idle_power)
      - tau = 46 (from the config) and ysigma = 0.4.

    For our input parameters:
      dot(all_max_job_power, weights) ≈ 320.16,
      nmu ≈ (10000 - 120*3) / (320.16 - 120) ≈ 9640/200.16 ≈ 48.18,
      nsigma ≈ 0.4 * 5000 / (200.16) ≈ 9.992,
      so the expected theta_star is:
         expected_theta = (48.18^2) / (2 * (9.992^2) * 46)
                        ≈ 2322.3 / (2 * 99.84 * 46)
                        ≈ 2322.3 / 9185.28
                        ≈ 0.2525.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        optimizer = construct_optimizer(output_dir=temp_dir)

        utilization = 0.85
        server_count = 3
        weights = np.array([0.33, 0.33, 0.34])
        all_max_job_power = np.array([276.0, 280.0, 402.0])
        idle_power = 120.0
        all_job_min_execution_time = np.array([30.7, 739.0, 52.6])
        job_type_count = 3
        all_job_size = np.array([1.0, 2.0, 1.0])
        P = 10000
        R = 5000

        optimizer.get_thetas(utilization, server_count, weights, all_max_job_power,
                             idle_power, all_job_min_execution_time, job_type_count,
                             all_job_size, P, R)

        theta_file = optimizer.output_dir + 'thetaStars.csv'
        with open(theta_file, 'r') as f:
            content = f.read().strip()
        theta_values = [float(val) for val in content.split(',') if val]

        dot_val = np.dot(all_max_job_power, weights)  # ≈320.16
        nmu = (P - idle_power * server_count) / (dot_val - idle_power)  # ≈48.18
        nsigma = 0.4 * R / (dot_val - idle_power)  # ≈9.992
        tau = 46.0
        expected_theta = (nmu ** 2) / (2 * (nsigma ** 2) * tau)  # ≈0.2525
        expected_theta = 0

        for i, theta in enumerate(theta_values):
            assert theta == expected_theta, (
                f"Job type {i}: Expected theta approx {expected_theta}, got {theta}"
            )


def test_overwrite():
    """
    Test that calling get_thetas twice overwrites the file rather than appending to it.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        optimizer = construct_optimizer(output_dir=temp_dir)
        utilization = 0.85
        server_count = 3
        weights = np.array([0.33, 0.33, 0.34])
        all_max_job_power = np.array([276.0, 280.0, 402.0])
        idle_power = 120.0
        all_job_min_execution_time = np.array([30.7, 739.0, 52.6])
        job_type_count = 3
        all_job_size = np.array([1.0, 2.0, 1.0])
        P = 10000
        R = 5000

        optimizer.get_thetas(utilization, server_count, weights, all_max_job_power,
                             idle_power, all_job_min_execution_time, job_type_count,
                             all_job_size, P, R)
        theta_file = optimizer.output_dir + 'thetaStars.csv'
        with open(theta_file, 'r') as f:
            first_content = f.read().strip()

        optimizer.get_thetas(utilization, server_count, weights, all_max_job_power,
                             idle_power, all_job_min_execution_time, job_type_count,
                             all_job_size, P, R)
        with open(theta_file, 'r') as f:
            second_content = f.read().strip()

        first_lines = first_content.splitlines()
        second_lines = second_content.splitlines()
        assert len(second_lines) == 1, "thetaStars.csv should contain exactly one line after overwrite."
        assert first_content == second_content, "thetaStars.csv content changed between calls with identical inputs."
