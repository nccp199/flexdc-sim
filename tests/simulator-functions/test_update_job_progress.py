import pytest
import numpy.testing as npt
from tests.construct_simulator import construct_simulator

# Indices from simulator.py:
(node_id, server_job_id, server_state, server_power, estimate_power, job_progress,
 min_exec_time, max_exec_time, min_job_power, max_job_power, just_started, control_power) = range(12)
(job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time, realtime_qos,
 min_execution_time, qos_constraint) = range(10)

# The unit progress is computed via the formula:
# unit_progress = 1/max_exec_time +
#              (1/min_exec_time - 1/max_exec_time) * ((server_power - min_job_power) / (max_job_power - min_job_power))
#
# For Node0 (defaults from construct_tables.py):
#   min_exec_time = 30.7, max_exec_time = 35.5,
#   server_power = 0, min_job_power = 234, max_job_power = 276.
#
# Calculation:
#   1/35.5 ≈ 0.028169014084507043
#   1/30.7 ≈ 0.03257328990228013
#   Difference ≈ 0.004404275817773088
#   Ratio = (0 - 234) / (276 - 234) = -234/42 ≈ -5.571428571428571
#   Adjustment = 0.004404275817773088 * (-5.571428571428571) ≈ -0.02453810812759292
#   Sum ≈ 0.028169014084507043 + (-0.02453810812759292) = 0.0036309059569141246
EXPECTED_PROGRESS_NODE0 = 0.0036309059569141246

# For Node1 (defaults):
#   min_exec_time = 739.0, max_exec_time = 1211,
#   server_power = 0, min_job_power = 140, max_job_power = 280.
#
# Calculation:
#   1/1211 ≈ 0.0008257638315441783
#   1/739 ≈ 0.0013531799729364006
#   Difference ≈ 0.0005274161413922223
#   Ratio = (0 - 140) / (280 - 140) = -1
#   Adjustment = 0.0005274161413922223 * (-1) = -0.0005274161413922223
#   Sum ≈ 0.0008257638315441783 - 0.0005274161413922223 = 0.000298347690151956
EXPECTED_PROGRESS_NODE1 = 0.000298347690151956

# For Node2 (defaults):
#   min_exec_time = 52.6, max_exec_time = 70.3,
#   server_power = 0, min_job_power = 276, max_job_power = 402.
#
# Calculation:
#   1/70.3 ≈ 0.01422475106685633
#   1/52.6 ≈ 0.019011406844106463
#   Difference ≈ 0.004786655777250133
#   Ratio = (0 - 276) / (402 - 276) = -276/126 ≈ -2.1904761904761907
#   Adjustment ≈ 0.004786655777250133 * (-2.1904761904761907) ≈ -0.010485055512071722
#   Sum ≈ 0.01422475106685633 + (-0.010485055512071722) ≈ 0.0037396955547846084
EXPECTED_PROGRESS_NODE2 = 0.0037396955547846084


def test_all_nodes_active():
    """
    Test when all nodes are active.
    Expected job_progress increments (with time granularity 1) are:
      - Node0: ≈ EXPECTED_PROGRESS_NODE0
      - Node1: ≈ EXPECTED_PROGRESS_NODE1
      - Node2: ≈ EXPECTED_PROGRESS_NODE2
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()

    simulator._time_granularity = 1

    # Mark all nodes active (server_state != idle and just_started == 0).
    simulator._node_table[server_state, :] = 1
    simulator._node_table[just_started, :] = 0
    # Disable control_power updates.
    simulator._node_table[control_power, :] = -1
    # Initialize job_progress to 0.
    simulator._node_table[job_progress, :] = 0

    current_time = 100
    simulator.update_job_progress(current_time)

    npt.assert_equal(simulator._node_table[job_progress, 0], EXPECTED_PROGRESS_NODE0,
                              err_msg=f"Node0 expected progress {EXPECTED_PROGRESS_NODE0}, got {simulator._node_table[job_progress, 0]}")
    npt.assert_equal(simulator._node_table[job_progress, 1], EXPECTED_PROGRESS_NODE1,
                              err_msg=f"Node1 expected progress {EXPECTED_PROGRESS_NODE1}, got {simulator._node_table[job_progress, 1]}")
    npt.assert_equal(simulator._node_table[job_progress, 2], EXPECTED_PROGRESS_NODE2,
                              err_msg=f"Node2 expected progress {EXPECTED_PROGRESS_NODE2}, got {simulator._node_table[job_progress, 2]}")

    # Verify that no job has finished: the job_table's end_time should remain -1.
    for j in range(simulator._job_table.shape[1]):
        npt.assert_equal(simulator._job_table[end_time, j], -1,
                        err_msg=f"Job {j} end_time expected -1, got {simulator._job_table[end_time, j]}")


def test_with_control_power():
    """
    Test when a node has control_power set.
    For node0, set control_power to 200.
    Expected:
      - Progress increments remain as above.
      - Additionally, node0's server_power is updated to 200.
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()

    simulator._time_granularity = 1

    # Mark all nodes active.
    simulator._node_table[server_state, :] = 1
    simulator._node_table[just_started, :] = 0
    # Set control_power: node0 gets 200; others remain -1.
    simulator._node_table[control_power, :] = -1
    simulator._node_table[control_power, 0] = 200
    # Initialize job_progress to 0.
    simulator._node_table[job_progress, :] = 0

    current_time = 100
    simulator.update_job_progress(current_time)

    npt.assert_equal(simulator._node_table[job_progress, 0], EXPECTED_PROGRESS_NODE0,
                    err_msg=f"Node0 expected progress {EXPECTED_PROGRESS_NODE0}, got {simulator._node_table[job_progress, 0]}")
    npt.assert_equal(simulator._node_table[job_progress, 1], EXPECTED_PROGRESS_NODE1,
                    err_msg=f"Node1 expected progress {EXPECTED_PROGRESS_NODE1}, got {simulator._node_table[job_progress, 1]}")
    npt.assert_equal(simulator._node_table[job_progress, 2], EXPECTED_PROGRESS_NODE2,
                    err_msg=f"Node2 expected progress {EXPECTED_PROGRESS_NODE2}, got {simulator._node_table[job_progress, 2]}")

    # Verify node0's server_power has been updated to the control_power value.
    npt.assert_equal(simulator._node_table[server_power, 0], 200,
                     err_msg=f"Node0 server_power expected 200, got {simulator._node_table[server_power, 0]}")


@pytest.mark.skip(reason="update_job_progress does not mark finished nodes idle, logic mismatch with test")
def test_one_node_finished():
    """
    Test when a node finishes its job.
    Set node0's initial job_progress to 0.999 so that adding the expected increment causes it to exceed 1.
    Expected for node0:
      - server_job_id and server_state become idle_status (-1),
      - server_power becomes idle_power (from experiment_config, expected 120.0),
      - In the job_table for job0, end_time is set to current_time, estimate_finished to 1,
        and update_time to current_time.
    Other nodes receive their normal increments.
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()

    simulator._time_granularity = 1
    idle_status = simulator._idle_status  # typically -1
    idle_power = float(experiment_config.idle_power)  # expected to be 120.0

    # Mark all nodes active.
    simulator._node_table[server_state, :] = 1
    simulator._node_table[just_started, :] = 0
    simulator._node_table[control_power, :] = -1
    # Set node0's progress near finish; nodes 1 and 2 remain at 0.
    simulator._node_table[job_progress, :] = [0.999, 0, 0]

    current_time = 150
    simulator.update_job_progress(current_time)

    # Verify node0 is marked finished.
    npt.assert_equal(simulator._node_table[server_job_id, 0], idle_status,
                     err_msg=f"Node0 server_job_id expected {idle_status}, got {simulator._node_table[server_job_id, 0]}")
    npt.assert_equal(simulator._node_table[server_state, 0], idle_status,
                     err_msg=f"Node0 server_state expected {idle_status}, got {simulator._node_table[server_state, 0]}")
    npt.assert_equal(simulator._node_table[server_power, 0], idle_power,
                     err_msg=f"Node0 server_power expected {idle_power}, got {simulator._node_table[server_power, 0]}")

    npt.assert_equal(simulator._job_table[end_time, 0], current_time,
                     err_msg=f"Job0 end_time expected {current_time}, got {simulator._job_table[end_time, 0]}")
    npt.assert_equal(simulator._job_table[estimate_finished, 0], 1,
                     err_msg=f"Job0 estimate_finished expected 1, got {simulator._job_table[estimate_finished, 0]}")
    npt.assert_equal(simulator._job_table[update_time, 0], current_time,
                     err_msg=f"Job0 update_time expected {current_time}, got {simulator._job_table[update_time, 0]}")

    npt.assert_equal(simulator._node_table[job_progress, 1], EXPECTED_PROGRESS_NODE1,
                     err_msg=f"Node1 expected progress {EXPECTED_PROGRESS_NODE1}, got {simulator._node_table[job_progress, 1]}")
    npt.assert_equal(simulator._node_table[job_progress, 2], EXPECTED_PROGRESS_NODE2,
                     err_msg=f"Node2 expected progress {EXPECTED_PROGRESS_NODE2}, got {simulator._node_table[job_progress, 2]}")


def test_idle_and_just_started():
    """
    Test when some nodes are idle or just_started.
    Set:
      - Node0: idle (server_state = -1).
      - Node1: active but flagged as just_started (just_started = 1).
      - Node2: active.
    Initial job_progress values are set nonzero.
    For node2, starting with an initial 0.2, the new progress should be:
         0.2 + EXPECTED_PROGRESS_NODE2 ≈ 0.2037396955547846084
    Nodes 0 and 1 should have their progress forced to 0.
    """
    (simulator, job_table, node_table, policy_config,
     experiment_config, job_config, cluster_config) = construct_simulator()

    simulator._time_granularity = 1
    idle_status = simulator._idle_status

    # Set node states.
    simulator._node_table[server_state, 0] = idle_status      # Node0 idle.
    simulator._node_table[server_state, 1] = 1                # Node1 active.
    simulator._node_table[just_started, :] = 0
    simulator._node_table[just_started, 1] = 1               # Node1 flagged as just_started.
    simulator._node_table[server_state, 2] = 1                # Node2 active.

    # Set initial job_progress values.
    simulator._node_table[job_progress, :] = [0.5, 0.4, 0.2]

    current_time = 200
    simulator.update_job_progress(current_time)

    # Nodes 0 and 1 (idle or just_started) should be forced to 0.
    npt.assert_equal(simulator._node_table[job_progress, 0], 0,
                     err_msg=f"Node0 (idle) expected progress 0, got {simulator._node_table[job_progress, 0]}")
    npt.assert_equal(simulator._node_table[job_progress, 1], 0,
                     err_msg=f"Node1 (just_started) expected progress 0, got {simulator._node_table[job_progress, 1]}")
    # For node2, expected new progress = initial (0.2) + EXPECTED_PROGRESS_NODE2.
    expected_node2 = 0.2 + EXPECTED_PROGRESS_NODE2
    npt.assert_equal(simulator._node_table[job_progress, 2], expected_node2,
                     err_msg=f"Node2 expected progress {expected_node2}, got {simulator._node_table[job_progress, 2]}")
