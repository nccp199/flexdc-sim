from tests.construct_aqa import construct_aqa
import numpy as np

(node_id, server_job_id, server_state, server_power, estimate_power, job_progress, min_exec_time, max_exec_time,
 min_job_power, max_job_power, just_started, control_power) = range(12)

def test_adjust_server_power_none_active():
    """
    No active servers (power should remain unchanged)
    """
    node_table = np.array([
        [-1, -1, -1, 0, 100, -1, -1, -1, 50, 150, -1, 0],
        [-1, -1, -1, 0, 100, -1, -1, -1, 50, 150, -1, 0],
        [-1, -1, -1, 0, 100, -1, -1, -1, 50, 150, -1, 0]
    ]).T
    aqa = construct_aqa(node_table)
    aqa.idle_status = -1
    aqa.adjust_server_power(20)
    assert (aqa._node_table[control_power] == np.array([0, 0, 0])).all()
    assert (aqa._node_table[estimate_power] == np.array([100, 100, 100])).all()

def test_adjust_server_power_reduction():
    """
    Power reduction needed (delta_power > 0)
    The power caps of active servers should be reduced proportionally
    """
    node_table = np.array([
        [-1, -1, 1, 0, 120, -1, -1, -1, 100, 150, -1, 0],
        [-1, -1, 2, 0, 140, -1, -1, -1, 110, 160, -1, 0],
        [-1, -1, 3, 0, 160, -1, -1, -1, 120, 170, -1, 0]
    ]).T
    aqa = construct_aqa(node_table)
    aqa.idle_status = -1
    aqa.adjust_server_power(30)
    assert (aqa._node_table[control_power] == np.array([140, 150, 160])).all()
    assert (aqa._node_table[estimate_power] == np.array([140, 150, 160])).all()

def test_adjust_server_power_no_delta():
    """
    No power reduction needed (delta_power = 0)
    control_power and estimate_power are set to max_job_power
    """
    node_table = np.array([
        [-1, -1, 1, 0, 120, -1, -1, -1, 100, 150, -1, 0],
        [-1, -1, 2, 0, 140, -1, -1, -1, 110, 160, -1, 0],
        [-1, -1, 3, 0, 160, -1, -1, -1, 120, 170, -1, 0]
    ]).T
    aqa = construct_aqa(node_table)
    aqa.idle_status = -1
    aqa.adjust_server_power(0)
    assert (aqa._node_table[control_power] == np.array([150, 160, 170])).all()
    assert (aqa._node_table[estimate_power] == np.array([150, 160, 170])).all()

def test_adjust_server_power_exceeds():
    """
    delta_power exceeds max_reducible_power (All servers set to min_job_power)
    """
    node_table = np.array([
        [-1, -1, 1, 0, 120, -1, -1, -1, 100, 150, -1, 120],
        [-1, -1, 2, 0, 140, -1, -1, -1, 110, 160, -1, 140],
        [-1, -1, 3, 0, 160, -1, -1, -1, 120, 170, -1, 160]
    ]).T
    aqa = construct_aqa(node_table)
    aqa.idle_status = -1
    aqa.adjust_server_power(160)
    assert (aqa._node_table[control_power] == np.array([100, 110, 120])).all()
    assert (aqa._node_table[estimate_power] == np.array([100, 110, 120])).all()

def test_adjust_server_power_mixed():
    """
    Mixed active and idle servers, some are adjusted and some are not
    """
    node_table = np.array([
        [-1, -1, 1, 0, 120, -1, -1, -1, 100, 150, -1, 120],
        [-1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0],
        [-1, -1, 3, 0, 160, -1, -1, -1, 120, 170, -1, 160]
    ]).T
    aqa = construct_aqa(node_table)
    aqa.idle_status = -1
    aqa.adjust_server_power(33)
    assert (aqa._node_table[control_power] == np.array([133, 0, 153])).all()
    assert (aqa._node_table[estimate_power] == np.array([133, 0, 153])).all()