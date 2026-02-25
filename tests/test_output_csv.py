import os
import pytest
from peacsim.parsing import experiment_config_reader as experiment_parser
from peacsim.parsing import cluster_profile_reader as cluster_parser
from peacsim.parsing import policy_config_reader as policy_parser
from peacsim.parsing import job_profile_reader as job_parser
import random
import numpy as np

from peacsim.run_simulator import init_output
from peacsim.simulator import Simulator
from peacsim.AQA_runtime_policy import AQARuntimePolicy
from peacsim.denormalize_PR import denormalize_p_and_r_to_watts
from peacsim.create_tables import init_job_table, init_node_table

def normalize_csv(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def run_simulator():
    cluster_config = cluster_parser.ClusterProfileReader("test-configs/reference/cluster.ini")
    policy_config = policy_parser.PolicyConfigReader("test-configs/reference/policy.ini")
    experiment_config = experiment_parser.ExperimentConfigReader("test-configs/reference/exp.ini")
    job_config = job_parser.JobProfileReader("test-configs/reference/W10.ini")
    np.random.seed(20)
    random.seed(20)
    job_table = init_job_table(experiment_config, job_config)
    node_table = init_node_table(experiment_config)
    P, R = denormalize_p_and_r_to_watts(experiment_config, job_config, policy_config.P_ratio, policy_config.R_ratio)
    policy_config._P = P
    policy_config._R = R
    runtime_policy = AQARuntimePolicy(policy_config, experiment_config, job_config, node_table, job_table)
    output_dir = "current_output/"
    simulator = Simulator(job_table, node_table, runtime_policy, experiment_config, job_config, cluster_config, output_dir)
    init_output(output_dir, experiment_config, policy_config)
    simulator.run()

@pytest.mark.skip(reason="test-configs/reference/ directory missing from repo")
def test_simulator_output_matches_reference():
    run_simulator()

    generated_lines = normalize_csv("reference_output/power_trace.csv")
    reference_lines = normalize_csv("current_output/power_trace.csv")

    assert len(generated_lines) == len(reference_lines), (
        f"Line count mismatch: generated={len(generated_lines)}, reference={len(reference_lines)}"
    )

    for i, (gen_line, ref_line) in enumerate(zip(generated_lines, reference_lines)):
        assert gen_line == ref_line, (
            f"Mismatch at line {i + 1}:\nGenerated: {gen_line}\nReference: {ref_line}"
        )