import numpy.testing as npt
from tests.construct_simulator import construct_simulator

(job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time, realtime_qos,
 min_execution_time, qos_constraint) = range(10)


def test_all_jobs_started():
    """
    Test update_realtime_qos() when all jobs are considered "started."

    Since default start_time is 0 (and _idle_status is –1), none of the jobs qualify for having
    update_time set to t. Therefore, update_time remains [0, 0, 0] and realtime_qos is computed as:
      realtime_qos = ((update_time – arrival_time) / min_execution_time) – estimate_finished.

    With default arrival_time = [0, 0, 0] and update_time = [0, 0, 0], realtime_qos becomes [0, 0, 0].
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()

    # Leave start_time unchanged (default 0, so not equal to _idle_status which is –1)
    simulator._job_table[start_time] = [0.0, 0.0, 0.0]
    # Ensure arrival_time is set (default is [0, 0, 0]; can change these if desired)
    simulator._job_table[arrival_time] = [0.0, 0.0, 0.0]
    # The update_time is initially [0, 0, 0]
    t = 100.0
    simulator.update_realtime_qos(t)

    # Expected: update_time remains [0, 0, 0] because no job has start_time == –1.
    npt.assert_equal(simulator._job_table[update_time].tolist(), [0.0, 0.0, 0.0],
                     err_msg="When all jobs are started, update_time should remain unchanged.")
    # realtime_qos = ((0 – 0)/min_execution_time) – 0 = 0 for all jobs.
    npt.assert_equal(simulator._job_table[realtime_qos].tolist(), [0.0, 0.0, 0.0],
                     err_msg="When all jobs are started, realtime_qos should be 0 for all jobs.")


def test_some_jobs_not_started():
    """
    Test update_realtime_qos() when some jobs have not started.

    We modify the job_table so that:
      - For job0: start_time is set to _idle_status (–1) and arrival_time is set to 10
      - For job1: start_time remains 0 (started) and arrival_time is set to 5
      - For job2: start_time is set to _idle_status (–1) and arrival_time is set to 20

    Then, with t = 100.0, the update will set update_time to 100.0 for jobs with start_time == –1
    and arrival_time < 100 (job0 and job2)

    The realtime_qos is then computed as:
      realtime_qos = ((update_time – arrival_time) / min_execution_time) – estimate_finished

    For job0:
      update_time becomes 100, so:
      realtime_qos = (100 – 10) / 30.7 – 0 = 90/30.7
      90/30.7 ≈ 2.9315960912052117

    For job1:
      update_time remains 0, so:
      realtime_qos = (0 – 5) / 739.0 – 0 = –5/739.0
      –5/739.0 ≈ –0.006765899864682003

    For job2:
      update_time becomes 100, so:
      realtime_qos = (100 – 20) / 52.6 – 0 = 80/52.6
      80/52.6 ≈ 1.520912547528517
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()

    idle_status = simulator._idle_status

    # Set start_time: job0 and job2 have not started; job1 has started
    simulator._job_table[start_time] = [idle_status, 0.0, idle_status]
    # Set arrival_time as specified.
    simulator._job_table[arrival_time] = [10.0, 5.0, 20.0]
    # Set estimate_finished to 0 for simplicity.
    simulator._job_table[estimate_finished] = [0.0, 0.0, 0.0]
    # Use the default min_execution_time from construct_tables: [30.7, 739.0, 52.6]

    # Initially, update_time is [0, 0, 0]
    t = 100.0
    simulator.update_realtime_qos(t)

    # Expected update_time: for jobs that have start_time == -1 and arrival_time < 100
    # That is, job0 and job2 get updated to 100.0; job1 remains 0
    expected_update_time = [100.0, 0.0, 100.0]
    npt.assert_equal(simulator._job_table[update_time].tolist(), expected_update_time,
                     err_msg="update_time not set correctly for not-started jobs.")

    EXPECTED_RTQ_JOB0 = 90 / 30.7  # ≈ 2.9315960912052117
    EXPECTED_RTQ_JOB1 = -5 / 739.0  # ≈ -0.006765899864682003
    EXPECTED_RTQ_JOB2 = 80 / 52.6  # ≈ 1.520912547528517
    expected_realtime_qos = [EXPECTED_RTQ_JOB0, EXPECTED_RTQ_JOB1, EXPECTED_RTQ_JOB2]

    npt.assert_equal(simulator._job_table[realtime_qos].tolist(), expected_realtime_qos,
                     err_msg="realtime_qos not computed as expected for not-started jobs.")


def test_with_estimate_finished():
    """
    Test update_realtime_qos() with nonzero estimate_finished values.

    We modify the job_table so that:
      - For job0: start_time = _idle_status (–1), arrival_time = 10, estimate_finished = 0.5, min_execution_time = 30.7.
        Then update_time becomes 100, and:
          realtime_qos = (100 – 10)/30.7 – 0.5 = 90/30.7 – 0.5 ≈ 2.9315960912052117 – 0.5 = 2.4315960912052117.
      - For job1: start_time remains 0 (started), arrival_time = 0, estimate_finished = 0.3, min_execution_time = 739.0.
        update_time remains 0, so realtime_qos = (0 – 0)/739.0 – 0.3 = –0.3.
      - For job2: start_time = _idle_status (–1), arrival_time = 20, estimate_finished = 1.0, min_execution_time = 52.6.
        Then update_time becomes 100, and:
          realtime_qos = (100 – 20)/52.6 – 1.0 = 80/52.6 – 1.0 ≈ 1.520912547528517 – 1.0 = 0.520912547528517.
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()
    idle_status = simulator._idle_status

    # Set up job0, job1, job2 as described.
    simulator._job_table[start_time] = [idle_status, 0.0, idle_status]
    simulator._job_table[arrival_time] = [10.0, 0.0, 20.0]
    simulator._job_table[estimate_finished] = [0.5, 0.3, 1.0]
    # Use default min_execution_time: [30.7, 739.0, 52.6]

    t = 100.0
    simulator.update_realtime_qos(t)

    expected_update_time = [100.0, 0.0, 100.0]
    npt.assert_equal(simulator._job_table[update_time].tolist(), expected_update_time,
                     err_msg="update_time not set correctly with nonzero estimate_finished.")

    expected_rtq_job0 = (100 - 10) / 30.7 - 0.5  # ≈ 2.4315960912052117
    expected_rtq_job1 = (0 - 0) / 739.0 - 0.3  # = -0.3
    expected_rtq_job2 = (100 - 20) / 52.6 - 1.0  # ≈ 0.520912547528517
    expected_realtime_qos = [expected_rtq_job0, expected_rtq_job1, expected_rtq_job2]

    npt.assert_equal(simulator._job_table[realtime_qos].tolist(), expected_realtime_qos,
                     err_msg="realtime_qos not computed correctly with nonzero estimate_finished.")


def test_arrival_equals_t():
    """
    Test update_realtime_qos() when a job's arrival_time equals t.

    For this test, set job0 with:
      - start_time = _idle_status (–1), arrival_time = 100, estimate_finished = 0, min_execution_time = 30.7.
    Since arrival_time is not less than t (t = 100), update_time remains 0,
    and realtime_qos = ((0 – 100) / 30.7) – 0 = -100/30.7
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()
    idle_status = simulator._idle_status

    simulator._job_table[start_time] = [idle_status, 0.0, 0.0]  # job0 not started, job1 and job2 started.
    simulator._job_table[arrival_time] = [100.0, 0.0, 0.0]  # job0 arrival equals t.
    simulator._job_table[estimate_finished] = [0.0, 0.0, 0.0]
    # Use default min_execution_time: for job0, 30.7.
    t = 100.0
    simulator.update_realtime_qos(t)

    # Since for job0, arrival_time == t, the condition arrival_time < t fails,
    # so update_time remains 0.
    expected_update_time = [0.0, 0.0, 0.0]
    npt.assert_equal(simulator._job_table[update_time].tolist(), expected_update_time,
                     err_msg="update_time should remain unchanged for a job with arrival_time equal to t.")

    # For job0: realtime_qos = (0 - 100)/30.7 - 0 = -100/30.7
    expected_rtq_job0 = -100 / 30.7
    # For job1 and job2, defaults yield realtime_qos = 0
    expected_realtime_qos = [expected_rtq_job0, 0.0, 0.0]
    npt.assert_equal(simulator._job_table[realtime_qos].tolist(), expected_realtime_qos,
                     err_msg="realtime_qos not computed correctly when arrival_time equals t.")


def test_no_jobs_arrived():
    """
    Test update_realtime_qos() when no jobs have arrived yet (arrival_time > t for all jobs).

    Set arrival_time for all jobs to 150 (greater than t = 100). Then no job qualifies for an update.
    Thus, update_time remains unchanged ([0, 0, 0]), and realtime_qos is computed as:
      realtime_qos = ((0 - arrival_time) / min_execution_time) - estimate_finished.

    For job0: realtime_qos = (0 - 150)/30.7 - 0 = -150/30.7
    For job1: realtime_qos = (0 - 150)/739.0 - 0 = -150/739.0
    For job2: realtime_qos = (0 - 150)/52.6 - 0 = -150/52.6
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()

    # Set arrival_time for all jobs to 150.
    simulator._job_table[arrival_time] = [150.0, 150.0, 150.0]
    # Set start_time for all jobs to not started (to potentially qualify), i.e. _idle_status.
    simulator._job_table[start_time] = [simulator._idle_status] * 3
    # Set estimate_finished to 0.
    simulator._job_table[estimate_finished] = [0.0, 0.0, 0.0]

    t = 100.0
    simulator.update_realtime_qos(t)

    # Since arrival_time (150) is not less than t (100), update_time remains unchanged.
    expected_update_time = [0.0, 0.0, 0.0]
    npt.assert_equal(simulator._job_table[update_time].tolist(), expected_update_time,
                     err_msg="update_time should remain unchanged when no job has arrived (arrival_time > t).")

    expected_rtq_job0 = (0 - 150) / 30.7  # -150/30.7
    expected_rtq_job1 = (0 - 150) / 739.0  # -150/739.0
    expected_rtq_job2 = (0 - 150) / 52.6  # -150/52.6
    expected_realtime_qos = [expected_rtq_job0, expected_rtq_job1, expected_rtq_job2]
    npt.assert_equal(simulator._job_table[realtime_qos].tolist(), expected_realtime_qos,
                     err_msg="realtime_qos not computed correctly when no job has arrived.")
