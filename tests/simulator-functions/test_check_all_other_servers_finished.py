import pytest
import numpy as np
from tests.construct_simulator import construct_simulator

# Use same indices as defined in simulator.py:
(node_id, server_job_id, server_state, server_power, estimate_power, job_progress,
 min_exec_time, max_exec_time, min_job_power, max_job_power, just_started, control_power) = range(12)

(job_id, job_type_id, arrival_time, start_time, end_time, estimate_finished, update_time, realtime_qos,
 min_execution_time, qos_constraint) = range(10)


def test_all_servers_finished():
    """
    Test when all servers have finished their jobs.

    We set job_progress for all servers to 1.
    Expected:
      - finished_servers should have exactly 3 entries.
      - finished_jobs should have exactly 3 entries.
    """
    simulator = (simulator, job_table, node_table, policy_config, experiment_config, job_config,
     cluster_config) = construct_simulator()
    # Mark all servers as finished by setting job_progress to 1.
    simulator._node_table[job_progress, :] = 1

    finished_servers, finished_jobs = simulator.check_all_other_servers_finished()

    # Explicit expected values: 3 servers and 3 jobs.
    assert len(finished_servers) == 3, f"Expected 3 finished servers, got {len(finished_servers)}"
    assert len(finished_jobs) == 3, f"Expected 3 finished jobs, got {len(finished_jobs)}"


def test_some_servers_not_finished():
    """
    Test when only one server has finished its job.

    We set job_progress to 1 only for the first server and ensure its server_job_id is 0.
    Expected:
      - finished_servers should have exactly 1 entry.
      - finished_jobs should have exactly 1 entry.
    """
    (simulator, job_table, node_table, policy_config, experiment_config, job_config,
     cluster_config) = construct_simulator()
    # Mark only the first server as finished.
    simulator._node_table[job_progress, 0] = 1
    # Ensure its job id is 0.
    simulator._node_table[server_job_id, 0] = 0
    # The remaining servers are left at job_progress 0.

    finished_servers, finished_jobs = simulator.check_all_other_servers_finished()

    # Explicit expected values: 1 finished server and 1 finished job.
    assert len(finished_servers) == 1, f"Expected 1 finished server, got {len(finished_servers)}"
    assert len(finished_jobs) == 1, f"Expected 1 finished job, got {len(finished_jobs)}"


def test_no_servers_finished():
    """
    Test when no servers have finished their jobs.

    We set job_progress for all servers to 0.
    Expected:
      - finished_servers and finished_jobs should both be empty.
    """
    (simulator, job_table, node_table, policy_config, experiment_config, job_config,
     cluster_config) = construct_simulator()
    # Mark all servers as not finished.
    simulator._node_table[job_progress, :] = 0

    finished_servers, finished_jobs = simulator.check_all_other_servers_finished()

    # Explicit expected values: 0 finished servers and 0 finished jobs.
    assert len(finished_servers) == 0, f"Expected 0 finished servers, got {len(finished_servers)}"
    assert len(finished_jobs) == 0, f"Expected 0 finished jobs, got {len(finished_jobs)}"
