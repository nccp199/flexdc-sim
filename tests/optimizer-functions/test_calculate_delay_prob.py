import os
import math
import tempfile
import pandas as pd
import numpy as np
from tests.construct_optimizer import construct_optimizer


def test_empty_job_table():
    """
    When the job table is empty, no finished jobs exist.
    Expected outcome:
      - delay_prob: [0]
      - mean: {0: nan} (since np.mean([]) returns nan)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        optimizer = construct_optimizer(output_dir=tmp_dir)
        df = pd.DataFrame(columns=['job_type_id', 'arrival_time', 'end_time'])
        sim_hour = 3  # 3 hours simulation; then first_hour=1 and last_hour=2.
        J = 1
        QoS_constraint = [1.0]  # Threshold multiplier: 1.0
        min_exec_time = [300]  # Minimum execution time: 300 seconds

        mean, delay_prob = optimizer.calculate_delay_prob(
            fit=False, sim_hour=sim_hour, J=J, job_table=df,
            QoS_constraint=QoS_constraint, min_execution_time=min_exec_time
        )
        assert delay_prob == [0], f"Expected delay_prob [0] but got {delay_prob}"
        assert math.isnan(mean[0]), f"Expected mean[0] to be NaN but got {mean[0]}"


def test_finished_job_only():
    """
    With one finished job:
      - job_type_id = 0, arrival_time = 4000, end_time = 4500.
      - Sojourn time = 4500 - 4000 = 500.
      - Adjusted delay = 500 - 300 = 200.
      - Threshold = QoS_constraint[0] * min_exec_time[0] = 1.0 * 300 = 300.
      - Since 200 < 300, no delay penalty is triggered.
    Expected outcome:
      - delay_prob: [0]
      - mean: {0: nan} (since no alpha is computed)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        optimizer = construct_optimizer(output_dir=tmp_dir)
        df = pd.DataFrame({
            'job_type_id': [0],
            'arrival_time': [4000],
            'end_time': [4500]
        })
        sim_hour = 3
        J = 1
        QoS_constraint = [1.0]
        min_exec_time = [300]

        mean, delay_prob = optimizer.calculate_delay_prob(
            fit=False, sim_hour=sim_hour, J=J, job_table=df,
            QoS_constraint=QoS_constraint, min_execution_time=min_exec_time
        )
        assert delay_prob == [0], f"Expected delay_prob [0] but got {delay_prob}"
        assert math.isnan(mean[0]), f"Expected mean[0] to be NaN but got {mean[0]}"


def test_job_should_finish_only():
    """
    With one unfinished job that qualifies as "should finish":
      - job_type_id = 0, arrival_time = 4000, end_time = -1.
      - For sim_hour = 3, simulation end time = 3 * 3600 = 10800.
      - Delay = (10800 - 4000) - 300 = 6800 - 300 = 6500.
      - Threshold = 300.
      - Since 6500 > 300, the condition is met.
    Expected outcome:
      - delay_prob: [1.0] (because 1 - 0 = 1 is appended)
      - mean: {0: nan} (no alpha computed as CDF value 0 is not in [fitdown,fitup])
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        optimizer = construct_optimizer(output_dir=tmp_dir)
        df = pd.DataFrame({
            'job_type_id': [0],
            'arrival_time': [4000],
            'end_time': [-1]
        })
        sim_hour = 3
        J = 1
        QoS_constraint = [1.0]
        min_exec_time = [300]

        mean, delay_prob = optimizer.calculate_delay_prob(
            fit=False, sim_hour=sim_hour, J=J, job_table=df,
            QoS_constraint=QoS_constraint, min_execution_time=min_exec_time
        )
        assert delay_prob == [1.0], f"Expected delay_prob [1.0] but got {delay_prob}"
        assert math.isnan(mean[0]), f"Expected mean[0] to be NaN but got {mean[0]}"


def test_combined_finished_and_should_finish():
    """
    With two jobs for job type 0:
      - Finished job: arrival_time = 4000, end_time = 4500 → delay = 500 - 300 = 200.
      - Unfinished job: arrival_time = 5000, end_time = -1 → delay = (10800 - 5000) - 300 = 5800 - 300 = 5500.
      - Sorted delays: [200, 5500].
      - np.linspace(0,1,2) yields [0, 1]. The condition in the loop checks:
          For idx=0: 200 > 300? False.
          For idx=1: 5500 > 300 is True, so delay_prob becomes 1 - 1 = 0.
    Expected outcome:
      - delay_prob: [0]
      - mean: {0: nan} (no alpha computed)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        optimizer = construct_optimizer(output_dir=tmp_dir)
        df = pd.DataFrame({
            'job_type_id': [0, 0],
            'arrival_time': [4000, 5000],
            'end_time': [4500, -1]
        })
        sim_hour = 3
        J = 1
        QoS_constraint = [1.0]
        min_exec_time = [300]

        mean, delay_prob = optimizer.calculate_delay_prob(
            fit=False, sim_hour=sim_hour, J=J, job_table=df,
            QoS_constraint=QoS_constraint, min_execution_time=min_exec_time
        )
        assert delay_prob == [0], f"Expected delay_prob [0] but got {delay_prob}"
        assert math.isnan(mean[0]), f"Expected mean[0] to be NaN but got {mean[0]}"


def test_fit_true_no_alpha():
    """
    With fit=True and one finished job:
      - job_type_id = 0, arrival_time = 4000, end_time = 4500.
      - Adjusted delay = 500 - 300 = 200.
      - Threshold = 1.0 * 300 = 300, so 200 < 300.
      - np.linspace(0,1,1) = [0]; hence no CDF value is in [fitdown,fitup] ([0.85,0.95]).
    Expected outcome:
      - delay_prob: [0]
      - mean: {0: nan}
    Note: Because fit=True, the method attempts to read thetaStars.csv.
          We create it using the same concatenation as production.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        optimizer = construct_optimizer(output_dir=tmp_dir)
        theta_file = optimizer.output_dir + 'thetaStars.csv'
        with open(theta_file, 'w') as f:
            f.write("1.0\n")
        df = pd.DataFrame({
            'job_type_id': [0],
            'arrival_time': [4000],
            'end_time': [4500]
        })
        sim_hour = 3
        J = 1
        QoS_constraint = [1.0]
        min_exec_time = [300]

        mean, delay_prob = optimizer.calculate_delay_prob(
            fit=True, sim_hour=sim_hour, J=J, job_table=df,
            QoS_constraint=QoS_constraint, min_execution_time=min_exec_time
        )
        assert delay_prob == [0], f"Expected delay_prob [0] but got {delay_prob}"
        assert math.isnan(mean[0]), f"Expected mean[0] to be NaN but got {mean[0]}"


def test_fit_true_alpha_computation():
    """
    In this test we adjust parameters to force alpha computation with a predictable value.

    We set:
      - QoS_constraint = [0.02] so that threshold = 0.02 * 300 = 6 seconds.
      - Create 10 finished jobs (job_type 0) with constant adjusted delay:
          Let arrival_time = 4000 and set end_time = 4000 + 300 + 10 = 4310.
          Then for each job, delay = 4310 - 4000 - 300 = 10.
          Since 10 > threshold (6), the condition is met.
      - With 10 identical delays, sorted delays = [10, 10, ..., 10].
      - np.linspace(0,1,10) produces:
            [0, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889, 1.0].
      - The delay_prob is set when the first delay exceeding threshold is encountered.
        For idx=0, iprob=0; since 10 > 6, delay_prob is appended as 1 - 0 = 1.
      - For alpha computation, with fit=True, the method checks each iprob.
        It appends alpha for each iprob between fitdown and fitup, where fitdown and fitup come from
        the gradient config (default dummy values are 0.85 and 0.95).
        In our np.linspace array, the only iprob in [0.85, 0.95] is at index 8: 0.8889.
      - For that index, the computed alpha is:

            alpha = (1 - iprob) / exp(- (delay) * theta)
                  = (1 - 0.8888888888888888) / exp(-10 * 1.0)
                  = 0.11111111111111116 / exp(-10)
                  = 0.11111111111111116 / 4.5399929762484854e-05
                  ≈ 2447.385088311858

    Expected outcome:
      - delay_prob: [1]
      - mean: {0: approximately 2447.3850883118585}
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        optimizer = construct_optimizer(output_dir=tmp_dir)
        # Create thetaStars.csv as required.
        theta_file = optimizer.output_dir + 'thetaStars.csv'
        with open(theta_file, 'w') as f:
            f.write("1.0\n")
        # Set QoS_constraint to [0.02] to lower threshold.
        QoS_constraint = [0.02]  # Threshold = 0.02*300 = 6 seconds.
        min_exec_time = [300]
        # Create 10 finished jobs with constant adjusted delay = 10.
        df = pd.DataFrame({
            'job_type_id': [0] * 10,
            'arrival_time': [4000] * 10,
            # end_time = 4000 + 300 + 10 = 4310.
            'end_time': [4310] * 10
        })
        sim_hour = 3  # simulation end = 10800 seconds.
        J = 1

        mean, delay_prob = optimizer.calculate_delay_prob(
            fit=True, sim_hour=sim_hour, J=J, job_table=df,
            QoS_constraint=QoS_constraint, min_execution_time=min_exec_time
        )
        # Expected delay_prob: For the first index meeting condition, iprob=0 so delay_prob = 1.
        expected_delay_prob = [1]
        assert delay_prob == expected_delay_prob, f"Expected delay_prob {expected_delay_prob} but got {delay_prob}"

        # Expected alpha at index 8:
        #   alpha = (1 - 0.8889)/exp(-10) ≈ 2447.3850883118585
        expected_alpha = 2447.3850883118585
        computed_alpha = mean[0]
        assert computed_alpha == expected_alpha, f"Expected alpha {expected_alpha} but got {computed_alpha}"
