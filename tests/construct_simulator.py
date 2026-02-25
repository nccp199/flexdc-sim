import configparser
import numpy as np
from src.peacsim.parsing import experiment_config_reader, policy_config_reader, job_profile_reader, cluster_profile_reader
from src.peacsim.simulator import Simulator
from tests.construct_tables import construct_tables
import random
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR)

EXP_FILENAME = os.path.join(PROJECT_ROOT, "test-configs", "test-simulator", "test-exp.ini")
JOB_FILENAME = os.path.join(PROJECT_ROOT, "test-configs", "test-simulator", "test-job.ini")
POLICY_FILENAME = os.path.join(PROJECT_ROOT, "test-configs", "test-simulator", "test-policy.ini")

def construct_simulator():
    """
    Constructs a simulator object and the relevant config objects for other tests to use.
    """
    #EXP_FILENAME = '../test-configs/test-simulator/test-exp.ini'
    config = configparser.ConfigParser()

    config["system"] = {
        "time_granularity": "1",
        "random_seed": "20",
        "simulation_duration": "3600",
        "utilization": "0.85",
        "idle_watts": "120.0",
        "server_count": "3"
    }

    config["iso"] = {
        "normalize_iso_signal": "True",
        "randomize_iso_start": "False",
        "iso_signal_granularity": "4",
        "iso_file_path": os.path.join(PROJECT_ROOT, "../data/ISO24h.csv")
    }

    with open(EXP_FILENAME, "w") as configfile:
        config.write(configfile)
    experiment_config = experiment_config_reader.ExperimentConfigReader(EXP_FILENAME)

    #job_filename = '../test-configs/test-simulator/test-job.ini'
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

    with open(JOB_FILENAME, "w") as configfile:
        config.write(configfile)

    job_config = job_profile_reader.JobProfileReader(JOB_FILENAME)

    #POLICY_FILENAME = '../test-configs/test-simulator/test-policy.ini'
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
        "weights_path": "../test-configs/test-simulator/test-weights.csv"
    }

    with open(POLICY_FILENAME, "w") as configfile:
        config.write(configfile)

    policy_config = policy_config_reader.PolicyConfigReader(POLICY_FILENAME)

    node_table, job_table = construct_tables()

    cluster_config = type('ClusterConfig', (object,), {
        'dell_1_idle_watts': 91.9,
        'dell_2_idle_watts': 91.8,
        'dell_3_idle_watts': 90
    })()

    np.random.seed(experiment_config.random_seed)
    random.seed(experiment_config.random_seed)

    simulator = Simulator(job_table, node_table, policy_config, experiment_config, job_config, cluster_config,
                          output_dir='../../tests/simulator-functions/output')
    return simulator, job_table, node_table, policy_config, experiment_config, job_config, cluster_config
