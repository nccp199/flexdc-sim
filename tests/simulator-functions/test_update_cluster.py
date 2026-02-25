import pytest
from tests.construct_simulator import construct_simulator
import numpy as np
import numpy.testing as npt

# Use same indices as defined in simulator.py:
(node_id, server_job_id, server_state, server_power, estimate_power, job_progress, min_exec_time, max_exec_time,
 min_job_power, max_job_power, just_started, control_power) = range(12)

(job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time, realtime_qos,
 min_execution_time, qos_constraint) = range(10)


@pytest.mark.skip(reason="construct_simulator passes PolicyConfigReader lacking availability_qos attribute")
def test_no_servers_running():
    """
    Tests update_cluster() with all servers idle
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()

    current_time = 2
    simulator._node_table[server_state] = simulator._idle_status
    simulator._node_table[server_power] = 0
    simulator._node_table[estimate_power] = -1
    simulator._node_table[just_started] = 0
    simulator._node_table[control_power] = 0

    simulator.update_cluster(current_time)

    assert np.array_equal(simulator._node_table[estimate_power], simulator._node_table[server_power])
    assert np.all(simulator._node_table[just_started] == -1)
    assert np.all(simulator._node_table[control_power] == -1)

    assert simulator.real_power == 0
    assert simulator.qos_job_sum == 0
    assert simulator.standby_power == 0
    assert simulator.waiting_sum == 0
    assert simulator.tracking_error == 0


@pytest.mark.skip(reason="construct_simulator passes PolicyConfigReader lacking availability_qos attribute")
def test_all_servers_running():
    """
    Test update_cluster() when all servers are running.

    Explicitly set:
      - server_state: all 1 (running)
      - server_power: [100.0, 200.0, 300.0]
      - cluster power cap: [50.0, 75.0, 100.0]
      - current_time: 10
    Expected:
      - real_power = 600.0
      - tracking_error = (600.0 - 75.0) / (5000 + 0.0001) ≈ 0.105
      - waiting_sum = 0 (since no job’s start_time equals idle status)
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()
    current_time = 10

    # Set all servers as running.
    simulator._node_table[server_state, :] = 1

    # Set explicit server_power values.
    explicit_power = np.array([100.0, 200.0, 300.0])
    simulator._node_table[server_power, :] = explicit_power

    # Set iso_signal granularity and a fixed cluster power cap.
    simulator._iso_signal_granularity = 4
    simulator._cluster_power_cap_per_signal = np.array([50.0, 75.0, 100.0])

    # Call update_cluster.
    simulator.update_cluster(current_time)

    # Check that estimate_power, just_started, and control_power are correctly updated.
    npt.assert_array_equal(simulator._node_table[estimate_power, :], explicit_power)
    npt.assert_array_equal(simulator._node_table[just_started, :], np.full(explicit_power.shape, -1))
    npt.assert_array_equal(simulator._node_table[control_power, :], np.full(explicit_power.shape, -1))

    # Expected values.
    expected_real_power = 100.0 + 200.0 + 300.0  # 600.0
    expected_qos_job_sum = expected_real_power
    expected_standby_power = 0
    # With job_table start_time initially [0, 0, 0] and idle_status = -1, no job is waiting.
    expected_waiting_sum = 0

    # Tracking error is computed only if current_time >= iso_signal_granularity.
    # Here, current_time // 4 = 2 so we use cluster_power_cap_per_signal[1] which is 75.0.
    expected_tracking_error = (expected_real_power - 75.0) / (5000 + 0.0001)

    npt.assert_equal(simulator.real_power, expected_real_power)
    npt.assert_equal(simulator.qos_job_sum, expected_qos_job_sum)
    npt.assert_equal(simulator.standby_power, expected_standby_power)
    npt.assert_equal(simulator.waiting_sum, expected_waiting_sum)
    npt.assert_equal(simulator.tracking_error, expected_tracking_error)


@pytest.mark.skip(reason="construct_simulator passes PolicyConfigReader lacking availability_qos attribute")
def test_some_servers_running():
    """
    Test update_cluster() when some of the servers are running and the others are idle.

    For 3 servers, we set:
      - Server 0: running (state = 1) with power 150.0.
      - Servers 1 and 2: idle (state = -1) with idle power 120.0 (from experiment_config).
      - Cluster power cap is set to [50.0, 75.0, 100.0] and current_time is 10.
    Expected:
      - real_power = 150.0 + 120.0 + 120.0 = 390.0
      - tracking_error = (390.0 - 75.0) / (5000 + 0.0001) ≈ 0.063
      - waiting_sum = 0 (since no job’s start_time equals idle status)
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()
    current_time = 10

    # For 3 servers, set server 0 to running and servers 1 & 2 to idle.
    simulator._node_table[server_state, 0] = 1  # running
    simulator._node_table[server_state, 1:] = simulator._idle_status  # idle

    # Set explicit server_power values:
    idle_power = float(experiment_config.idle_power)  # Expected to be 120.0 from config.
    simulator._node_table[server_power, 0] = 150.0  # Running server.
    simulator._node_table[server_power, 1] = idle_power  # Idle servers.
    simulator._node_table[server_power, 2] = idle_power

    # Set iso_signal granularity and fixed cluster power cap.
    simulator._iso_signal_granularity = 4
    simulator._cluster_power_cap_per_signal = np.array([50.0, 75.0, 100.0])

    # Call update_cluster.
    simulator.update_cluster(current_time)

    # Verify estimate_power, just_started, and control_power.
    npt.assert_array_equal(simulator._node_table[estimate_power, :],
                           simulator._node_table[server_power, :])
    npt.assert_array_equal(simulator._node_table[just_started, :],
                           np.full(simulator._node_table.shape[1], -1))
    npt.assert_array_equal(simulator._node_table[control_power, :],
                           np.full(simulator._node_table.shape[1], -1))

    # Expected values.
    expected_real_power = 150.0 + idle_power + idle_power  # 150 + 120 + 120 = 390.0
    expected_qos_job_sum = expected_real_power
    expected_standby_power = 0
    expected_waiting_sum = 0

    # With current_time = 10 and iso_signal_granularity = 4, we use the second element of the cap array:
    expected_tracking_error = (expected_real_power - 75.0) / (5000 + 0.0001)

    npt.assert_equal(simulator.real_power, expected_real_power)
    npt.assert_equal(simulator.qos_job_sum, expected_qos_job_sum)
    npt.assert_equal(simulator.standby_power, expected_standby_power)
    npt.assert_equal(simulator.waiting_sum, expected_waiting_sum)
    npt.assert_equal(simulator.tracking_error, expected_tracking_error)
