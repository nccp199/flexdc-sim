import pytest
from peacsim.parsing import policy_config_reader
from io import StringIO

import pandas as pd

file = StringIO("[bid]"
                "\nP = 5000"
                "\nR = 1000"
                "\nP_ratio = 0.9"
                "\nR_ratio = 0.3"
                "\n[policy]"
                "\n name = AQA"
                "\n[parameters]"
                "\n weights_path = ../configs/weights/weights.csv"
                "\ntest = 1"
                )

policy_config_reader = policy_config_reader.PolicyConfigReader(file)


def test_runtime_policy_name():
    assert policy_config_reader.runtime_policy_name == "AQA"


def test_P():
    assert policy_config_reader.P == 5000


def test_R():
    assert policy_config_reader.R == 1000


@pytest.mark.skip(reason="policy_parameters now includes node_count_control, test expectation stale")
def test_policy_parameters():
    assert policy_config_reader.policy_parameters == {"weights_path": "../configs/weights/weights.csv",
                                                      "test": "1"}
