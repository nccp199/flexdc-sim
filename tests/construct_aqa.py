import configparser

from src.peacsim.AQA_runtime_policy import AQARuntimePolicy
from src.peacsim.parsing import experiment_config_reader, policy_config_reader, job_profile_reader
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR)


def construct_aqa(node_table=np.array([]), job_table=np.array([])):
    exp_filename = os.path.join(PROJECT_ROOT, "test-configs", "test-aqa", "test-exp.ini")

    config = configparser.ConfigParser()

    config["system"] = {
        "time_granularity": "1",
        "random_seed": "20",
        "simulation_duration": "3600",
        "utilization": "0.85",
        "idle_watts": "100.0",
        "server_count": "10"
    }

    config["iso"] = {
        "normalize_iso_signal": "True",
        "randomize_iso_start": "False",
        "iso_signal_granularity": "4",
        "iso_file_path": "../../data/ISO24h.csv"
    }

    with open(exp_filename, "w") as configfile:
        config.write(configfile)
    exp_config = experiment_config_reader.ExperimentConfigReader(exp_filename)

    job_filename = os.path.join(PROJECT_ROOT, "test-configs", "test-aqa", "test-job.ini")
    config = configparser.ConfigParser()

    config["DEFAULT"] = {
        "job_size": "1"
    }

    config["test-type-0"] = {
        "min_job_power_watts": "234",
        "max_job_power_watts": "276",
        "min_time_seconds": "30.7",
        "max_time_seconds": "35.3",
        "qos_constraint": "3.0"
    }

    config["test-type-1"] = {
        "min_job_power_watts": "140",
        "max_job_power_watts": "280",
        "min_time_seconds": "739",
        "max_time_seconds": "1211",
        "qos_constraint": "5.0",
        "job_size": "2"
    }

    config["test-type-2"] = {
        "min_job_power_watts": "276",
        "max_job_power_watts": "402",
        "min_time_seconds": "52.6",
        "max_time_seconds": "70.3",
        "qos_constraint": "4.7",
    }

    with open(job_filename, "w") as configfile:
        config.write(configfile)

    job_config = job_profile_reader.JobProfileReader(job_filename)

    policy_filename = os.path.join(PROJECT_ROOT, "test-configs", "test-aqa", "test-policy.ini")
    config = configparser.ConfigParser()

    config["bid"] = {
        "P": "10000",
        "R": "5000",
        "P_ratio": "0.9",
        "R_ratio": "0.3"
    }

    config["policy"] = {
        "name": "AQA"
    }

    config["parameters"] = {
        "weights_path": os.path.join(PROJECT_ROOT, "test-configs", "test-aqa", "test-weights.csv")
    }

    with open(policy_filename, "w") as configfile:
        config.write(configfile)

    policy_config = policy_config_reader.PolicyConfigReader(policy_filename)
    return AQARuntimePolicy(policy_config, exp_config, job_config, node_table, job_table)




