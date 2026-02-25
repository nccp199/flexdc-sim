import numpy.testing as npt
from tests.construct_simulator import construct_simulator

# Job table indices (as defined in simulator.py):
# (job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time,
#  realtime_qos, min_execution_time, qos_constraint) = range(10)
(job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time,
 realtime_qos, min_execution_time, qos_constraint) = range(10)


def test_all_jobs_started():
    """
    When all jobs are already started (start_time != _idle_status),
    no job is queued so that calc_qos_emergency() finds no emergency.
    Expected: qos_emergency should be False.
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()

    # For "started" jobs, set start_time to 0 (default) which is not equal to _idle_status (-1)
    simulator._job_table[start_time] = [0.0, 0.0, 0.0]
    # Set arrival_time to some values (all ≤ current_time) but it doesn't matter
    simulator._job_table[arrival_time] = [0.0, 10.0, 20.0]

    current_time = 100.0
    simulator.calc_qos_emergency(current_time)

    # Since no job is queued, we expect qos_emergency to be False.
    npt.assert_equal(simulator.qos_emergency, False,
                     err_msg="qos_emergency should be False when no job is queued.")


def test_no_emergency():
    """
    For job0, set it as queued:
      - start_time = -1 (queued)
      - arrival_time = 0
      - min_execution_time = 30.7
      - qos_constraint = 5.0
    Then boundary_start_time = 0 + (5.0 * 30.7) = 153.5.
    Let current_time = 100 (which is less than 153.5).
    Expected: No emergency → qos_emergency should be False.

    For the other jobs, we mark them as started.
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()
    idle_status = simulator._idle_status  # -1

    # Mark job0 as queued; jobs 1 and 2 as started.
    simulator._job_table[start_time] = [idle_status, 0.0, 0.0]
    # Set arrival times (job0 arrives at 0)
    simulator._job_table[arrival_time] = [0.0, 10.0, 20.0]
    # Set min_execution_time and qos_constraint for job0 to control deadline.
    # Using default for job0 is 30.7 and 3.0, but we override qos_constraint to 5.0.
    simulator._job_table[min_execution_time][0] = 30.7
    simulator._job_table[qos_constraint][0] = 5.0  # boundary = 0 + 5*30.7 = 153.5

    current_time = 100.0
    simulator.calc_qos_emergency(current_time)

    # Expected: current_time (100) < boundary (153.5), so no emergency.
    npt.assert_equal(simulator.qos_emergency, False,
                     err_msg="qos_emergency should be False when no queued job has passed its deadline.")


def test_one_emergency():
    """
    For job0, set it as queued:
      - start_time = -1 (queued)
      - arrival_time = 0
      - min_execution_time = 30.7
      - qos_constraint = 2.0
    Then boundary_start_time = 0 + (2.0 * 30.7) = 61.4.
    Let current_time = 100 (which is >= 61.4).
    Expected: Emergency is triggered.
      - start_emergency becomes True.
      - emergency_end = current_time + min_execution_time of job0 = 100 + 30.7 = 130.7.
      - qos_emergency = True (since emergency is True or 100 <= 130.7).
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()
    idle_status = simulator._idle_status  # -1

    # Mark job0 as queued; jobs 1 and 2 as started.
    simulator._job_table[start_time] = [idle_status, 0.0, 0.0]
    simulator._job_table[arrival_time] = [0.0, 10.0, 20.0]
    # Override job0's parameters:
    simulator._job_table[min_execution_time][0] = 30.7
    simulator._job_table[qos_constraint][0] = 2.0  # boundary = 0 + 2*30.7 = 61.4

    current_time = 100.0
    simulator.calc_qos_emergency(current_time)

    # Expected: qos_emergency should be True.
    npt.assert_equal(simulator.qos_emergency, True,
                     err_msg="qos_emergency should be True when a queued job has passed its deadline.")


def test_multiple_emergency():
    """
    Set all three jobs as queued by setting start_time = -1.
    Configure:
      - Job0: arrival_time = 0, min_execution_time = 30.7, qos_constraint = 5.0.
               boundary = 0 + 5*30.7 = 153.5.
      - Job1: arrival_time = 10, min_execution_time = 739.0, qos_constraint = 1.0.
               boundary = 10 + 739.0 = 749.0.
      - Job2: arrival_time = 20, min_execution_time = 52.6, qos_constraint = 2.0.
               boundary = 20 + 2*52.6 = 20 + 105.2 = 125.2.
    Let current_time = 130.

    Then for the three jobs:
      - Job0: 130 < 153.5 → not past deadline.
      - Job1: 130 < 749.0 → not past deadline.
      - Job2: 130 >= 125.2 → past deadline.
    Thus, start_emergency becomes True (triggered by job2).
    emergency_idx is the index corresponding to job2.
    emergency_end = current_time + min_execution_time for job2 = 130 + 52.6 = 182.6.
    qos_emergency is then True.
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()
    idle_status = simulator._idle_status  # -1

    # Mark all jobs as queued.
    simulator._job_table[start_time] = [idle_status, idle_status, idle_status]
    simulator._job_table[arrival_time] = [0.0, 10.0, 20.0]
    # Set parameters for each job:
    simulator._job_table[min_execution_time] = [30.7, 739.0, 52.6]
    simulator._job_table[qos_constraint] = [5.0, 1.0, 2.0]  # Boundaries: 153.5, 749.0, 125.2 respectively.

    current_time = 130.0
    simulator.calc_qos_emergency(current_time)

    # Expected: qos_emergency should be True (since job2 is overdue).
    npt.assert_equal(simulator.qos_emergency, True,
                     err_msg="qos_emergency should be True when at least one queued job has passed its deadline.")


def test_no_jobs_arrived():
    """
    Set all jobs as queued (start_time = -1) but with arrival_time greater than current_time.
    For example, set arrival_time for all jobs to 200 and current_time = 100.
    Then, queued_jobs is empty (because arrival_time <= current_time is False for all),
    so no emergency is detected.
    Expected: qos_emergency should be False.
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()

    # Mark all jobs as queued.
    simulator._job_table[start_time] = [-1.0, -1.0, -1.0]  # queued
    simulator._job_table[arrival_time] = [200.0, 200.0, 200.0]  # None has arrived by current_time = 100.

    current_time = 100.0
    simulator.calc_qos_emergency(current_time)

    npt.assert_equal(simulator.qos_emergency, False,
                     err_msg="qos_emergency should be False when no job has arrived (arrival_time > current_time).")


def test_borderline_start_time():
    """
    Test when current_time equals the boundary_start_time exactly for a queued job.

    For job0, set:
      - start_time = -1 (queued)
      - arrival_time = 0
      - min_execution_time = 30.7
      - qos_constraint such that boundary = 100.0.

    Solve: boundary = 0 + qos_constraint * 30.7 = 100  → qos_constraint = 100 / 30.7 ≈ 3.256 (exact value computed).

    Set qos_constraint to exactly 100/30.7.
    Then, with current_time = 100, we have:
      current_time >= boundary (100 >= 100) → emergency triggered.
      emergency_end = 100 + 30.7 = 130.7.
      qos_emergency = True.
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()
    idle_status = simulator._idle_status  # -1

    # Mark job0 as queued; others as started.
    simulator._job_table[start_time] = [idle_status, 0.0, 0.0]
    simulator._job_table[arrival_time] = [0.0, 10.0, 20.0]
    # Set job0's min_execution_time as default 30.7 and qos_constraint to exactly 100/30.7.
    simulator._job_table[min_execution_time][0] = 30.7
    qos_const = 100.0 / 30.7
    simulator._job_table[qos_constraint][0] = qos_const  # This makes boundary = 100.

    current_time = 100.0
    simulator.calc_qos_emergency(current_time)

    npt.assert_equal(simulator.qos_emergency, True,
                     err_msg="qos_emergency should be True when current_time equals the boundary_start_time.")

