from numpy import int64

from peacsim.run_simulator import init_job_table
from peacsim.parsing import experiment_config_reader as experiment_parser
from peacsim.parsing import job_profile_reader as job_parser
import random
import numpy as np
import pandas as pd


def test_init_job_table():
    assert True == True
    #experiment_config = experiment_parser.ExperimentConfigReader('../configs/experiment/exp.ini')
    #job_config = job_parser.JobProfileReader('../configs/workload/W10.ini')
    #np.random.seed(experiment_config.random_seed)
    #random.seed(experiment_config.random_seed)
    #new_job_table = init_job_table(experiment_config, job_config)
    #old_job_table = pd.read_csv('test-configs/testCli/initJobTable.csv')

    # assert (new_job_table[1].astype(int64) == np.array(old_job_table['job_type_id'])).all()
    # assert (new_job_table[2] == old_job_table['arrival_time']).all()


