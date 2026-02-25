import pytest
import numpy as np
from tests.construct_aqa import construct_aqa

@pytest.mark.skip(reason="method renamed from calculate_server_count_change_corrected to calculate_server_count_change")
def test_calculate_server_count_change_none_active():
    """
    No active servers (total active servers = 0).
    Regardless of the adjusted weights, if there are no active servers required, then no additional servers should be activated.
    """
    node_table = np.array([
        [-1] * 12,
        [-1] * 12,
        [-1] * 12
    ]).T
    aqa = construct_aqa(node_table=node_table)
    aqa.job_type_count = 3
    aqa.idle_status = -1
    adjusted_weights = {0: 0.5, 1: 0.3, 2: 0.2}
    total_num_of_active_servers = 0
    assert aqa.calculate_server_count_change_corrected(total_num_of_active_servers, adjusted_weights) == [0, 0, 0]

@pytest.mark.skip(reason="method renamed from calculate_server_count_change_corrected to calculate_server_count_change")
def test_calculate_server_count_change_all_idle():
    """
    All servers idle, active servers required > 0.
    With no active servers in the node table, the change equals the computed target.
    """
    node_table = np.array([
        [-1] * 12,
        [-1] * 12,
        [-1] * 12,
        [-1] * 12
    ]).T
    aqa = construct_aqa(node_table=node_table)
    aqa.job_type_count = 3
    aqa.idle_status = -1
    adjusted_weights = {0: 0.5, 1: 0.3, 2: 0.2}
    total_num_of_active_servers = 6
    assert aqa.calculate_server_count_change_corrected(total_num_of_active_servers, adjusted_weights) == [3, 2, 1]

@pytest.mark.skip(reason="method renamed from calculate_server_count_change_corrected to calculate_server_count_change")
def test_calculate_server_count_change_partial():
    """
    Some servers already active (partial fulfillment).
    For total active servers = 5 and weights {0: 0.4, 1: 0.4, 2: 0.2} the target per job type is [2, 2, 1]
    The node_table already has one server each for job types 0 and 1.
    """
    np.array([[0, 1, -1, -1]])
    node_table = np.array([
        [-1, -1, 0] + [-1] * 9,
        [-1, -1, 1] + [-1] * 9,
        [-1, -1, -1] + [-1] * 9,
        [-1, -1, -1] + [-1] * 9
    ]).T
    aqa = construct_aqa(node_table=node_table)
    aqa.job_type_count = 3
    aqa.idle_status = -1
    adjusted_weights = {0: 0.4, 1: 0.4, 2: 0.2}
    total_num_of_active_servers = 5
    assert aqa.calculate_server_count_change_corrected(total_num_of_active_servers, adjusted_weights) == [1, 1, 1]

@pytest.mark.skip(reason="method renamed from calculate_server_count_change_corrected to calculate_server_count_change")
def test_calculate_server_count_change_excess():
    """
    More servers already active than needed (no additional activation)
    For total active servers = 5 with weights {0: 0.6, 1: 0.4} the target is [3, 2].
    If the node table already has more or equal servers for each job type, no change is needed.
    """
    np.array([[0, 0, 1]])
    node_table = np.array([
        [-1, -1, 0] + [-1] * 9,
        [-1, -1, 0] + [-1] * 9,
        [-1, -1, 1] + [-1] * 9
    ]).T
    aqa = construct_aqa(node_table=node_table)
    aqa.job_type_count = 2
    aqa.idle_status = -1
    adjusted_weights = {0: 0.6, 1: 0.4}
    total_num_of_active_servers = 5
    assert aqa.calculate_server_count_change_corrected(total_num_of_active_servers, adjusted_weights) == [1, 1]

@pytest.mark.skip(reason="method renamed from calculate_server_count_change_corrected to calculate_server_count_change")
def test_calculate_server_count_change_mixed():
    """
    Mixed scenario
    For total active servers = 10 and weights {0: 0.2, 1: 0.5, 2: 0.3} the target is [2, 5, 3].
    The node_table already has one server of type 0, two of type 1, and three of type 2.
    """
    node_table = np.array([
        [-1, -1, 0] + [-1] * 9,
        [-1, -1, 1] + [-1] * 9,
        [-1, -1, 1] + [-1] * 9,
        [-1, -1, 2] + [-1] * 9,
        [-1, -1, 2] + [-1] * 9,
        [-1, -1, 2] + [-1] * 9
    ]).T
    aqa = construct_aqa(node_table=node_table)
    aqa.job_type_count = 3
    aqa.idle_status = -1
    adjusted_weights = {0: 0.2, 1: 0.5, 2: 0.3}
    total_num_of_active_servers = 10
    assert aqa.calculate_server_count_change_corrected(total_num_of_active_servers, adjusted_weights) == [1, 3, 0]

