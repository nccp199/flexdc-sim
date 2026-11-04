"""Exercise restoration through real policies and the existing actuator timing."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from peacsim.AQA_runtime_policy import AQARuntimePolicy as LegacyPolicy
from peacsim.aqa_runtimepolicy import AQARuntimePolicy
from peacsim.create_tables import generate_job_table, init_node_table
from peacsim.flex_resource_policy import FlexResourcePolicy
from peacsim.proportional_cap_policy import ProportionalCapPolicy
from peacsim.priority_cap_policy import PriorityCapPolicy
from peacsim.simulator import Simulator
from peacsim.node_table_dao import (
    server_job_id, server_state, server_power, estimate_power, job_progress,
    min_exec_time, max_exec_time, min_job_power, max_job_power, just_started,
    control_power,
)

POLICIES = [
    pytest.param((cls, name), id=f"{generation}-{name}")
    for generation, cls, name in [
        ("new", AQARuntimePolicy, "AQA"),
        ("new", ProportionalCapPolicy, "proportional-cap"),
        ("new", PriorityCapPolicy, "priority-cap"),
        ("new", FlexResourcePolicy, "flex-resource"),
        *[("legacy", LegacyPolicy, name) for name in
          ["AQA", "proportional-cap", "priority-cap", "flex-resource"]],
    ]
]


@pytest.fixture(params=POLICIES)
def system(request, tmp_path):
    cls, name = request.param
    exp = NS(server_count=2, idle_power=50.)
    jobs = NS(job_type_count=2, all_jobs={"first": 0, "second": 1},
              all_max_job_power={0: 200., 1: 200.},
              all_min_job_power={0: 100., 1: 100.},
              all_min_execution_time={0: 100., 1: 100.},
              all_max_execution_time={0: 200., 1: 200.},
              all_job_size={0: 1, 1: 1}, all_job_qos_constraints={0: 4., 1: 4.})
    weights = tmp_path / "weights.csv"
    weights.write_text("job_type_id,weights\n0,0.5\n1,0.5\n")
    config = NS(runtime_policy_name=name, P=250., R=100., policy_parameters={
        "weights_path": str(weights), "node_count_control": False,
        "power_cap_percentage": 1., "probabilities": [1., 1.]})
    table = generate_job_table(2, [0, 1], [0, 1000], jobs)
    table[3, 0] = 0  # The first job is already running.
    nodes = init_node_table(exp)
    nodes[control_power] = -1  # Simulator.update_cluster clears commands before scheduling.
    nodes[server_job_id, 0] = 0
    nodes[server_state, 0] = 0
    nodes[server_power, 0] = nodes[estimate_power, 0] = 150.
    nodes[min_exec_time, 0], nodes[max_exec_time, 0] = 100., 200.
    nodes[min_job_power, 0], nodes[max_job_power, 0] = 100., 200.
    nodes[just_started, 0] = -1
    policy = cls(config, exp, jobs, nodes, table)
    sim = Simulator.__new__(Simulator)
    sim._node_table = nodes
    sim._idle_status = -1
    sim._time_granularity = 1
    return policy, sim, nodes, table


def test_repeated_cap_release_preserves_actuator_timing(system):
    policy, sim, nodes, _ = system
    for step, (budget, expected) in enumerate(
        [(180, 180), (200, 200), (200, 200), (140, 140),
         (200, 200), (250, 200), (50, 100), (200, 200)], start=1
    ):
        previous_power = nodes[server_power, 0]
        previous_progress = nodes[job_progress, 0]
        nodes[control_power] = -1
        nodes[estimate_power] = nodes[server_power]
        policy.execute(step, budget + 50.)  # Include the idle server's power.
        assert nodes[control_power, 0] == pytest.approx(expected)
        assert nodes[estimate_power, 0] == pytest.approx(expected)
        assert nodes[server_power, 0] == previous_power
        assert nodes[control_power, 1] == -1
        sim.update_job_progress(step + 1)
        old_power_progress = 1 / 200 + (1 / 100 - 1 / 200) * ((previous_power - 100) / 100)
        assert nodes[job_progress, 0] == pytest.approx(previous_progress + old_power_progress)
        assert nodes[server_power, 0] == pytest.approx(expected)
        assert nodes[server_power, 1] == 50.


@pytest.mark.parametrize("target", [10., 500.])
def test_idle_servers_receive_no_command(system, target):
    policy, _, nodes, table = system
    nodes[server_job_id] = nodes[server_state] = -1
    nodes[server_power] = nodes[estimate_power] = 50.
    table[3] = -1
    table[2] = 1000
    policy.execute(1, target)
    np.testing.assert_array_equal(nodes[control_power], [-1, -1])
    np.testing.assert_array_equal(nodes[server_power], [50., 50.])


@pytest.mark.parametrize("target", [100., 500.])
def test_no_reducible_power_still_has_an_explicit_command(system, target):
    policy, _, nodes, _ = system
    nodes[min_job_power, 0] = nodes[max_job_power, 0] = 200.
    nodes[server_power, 0] = 200.
    policy.execute(1, target)
    assert nodes[control_power, 0] == 200.
    assert nodes[control_power, 1] == -1
    assert np.isfinite(nodes[control_power]).all()


@pytest.mark.parametrize("budget, expected", [(150., 150.), (200., 200.)])
def test_newly_scheduled_job_is_included(system, budget, expected):
    policy, sim, nodes, table = system
    nodes[server_job_id] = nodes[server_state] = -1
    nodes[server_power] = nodes[estimate_power] = 50.
    table[3, 0] = -1
    policy.execute(0, budget + 50.)
    assert nodes[server_job_id, 0] == 0
    assert nodes[control_power, 0] == pytest.approx(expected)
    assert nodes[server_power, 0] == 200.  # Native scheduler initialization.
    assert nodes[control_power, 1] == -1
    sim.update_job_progress(1)
    assert nodes[server_power, 0] == pytest.approx(expected)
    assert nodes[job_progress, 0] == 0  # Preserve just-started handling.
