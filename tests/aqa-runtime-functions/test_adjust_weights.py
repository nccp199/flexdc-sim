import numpy as np

from tests.construct_aqa import construct_aqa

(job_type_id, start_time, arrival_time) = (0, 1, 2)

def test_adjust_weights_none():
    """
    No Jobs Queued (All jobs have already started)
    """
    job_table = np.array([
        [-1, 1, 10, 5, -1, -1, -1, -1, -1, -1],
        [-1, 2, 12, 5, -1, -1, -1, -1, -1, -1]
    ]).T
    aqa = construct_aqa(job_table=job_table)
    aqa.weights = {1: 0.5, 2: 0.5}
    aqa.idle_status = -1
    assert aqa.adjust_weights(current_time=3) == {1: 0, 2: 0}

def test_adjust_weights_some():
    """
    Some Jobs Queued (Only job_type=1 is queued)
    """
    job_table = np.array([
        [-1, 1, 2, -1, -1, -1, -1, -1, -1, -1],
        [-1, 2, 3, 5, -1, -1, -1, -1, -1, -1]
    ]).T
    aqa = construct_aqa(job_table=job_table)
    aqa.weights = {1: 0.6, 2: 0.4}
    aqa.idle_status = -1
    assert aqa.adjust_weights(current_time=3) == {1: 1.0, 2: 0.0}

def test_adjust_weights_all():
    """
    All Job Types Queued
    """
    job_table = np.array([
        [-1, 1, 2, -1, -1, -1, -1, -1, -1, -1],
        [-1, 2, 3, -1, -1, -1, -1, -1, -1, -1]
    ]).T
    aqa = construct_aqa(job_table=job_table)
    aqa.weights = {1: 0.6, 2: 0.4}
    aqa.idle_status = -1
    assert aqa.adjust_weights(current_time=3) == {1: 0.6, 2: 0.4}

def test_adjust_weights_mixed():
    """
    Mixed Job Types in Queue
    """
    job_table = np.array([
        [-1, 1, 2, -1, -1, -1, -1, -1, -1, -1],
        [-1, 3, 3, -1, -1, -1, -1, -1, -1, -1],
        [-1, 2, 4, 5, -1, -1, -1, -1, -1, -1]
    ]).T
    aqa = construct_aqa(job_table=job_table)
    aqa.weights = {1: 0.5, 2: 0.3, 3: 0.2}
    aqa.idle_status = -1
    assert aqa.adjust_weights(current_time=3) == {1: 0.7142857142857143, 3: 0.28571428571428575, 2: 0}

def test_adjust_weights_single():
    """
    Single Job Type Queued (Only job_type=2 is queued)
    """
    job_table = np.array([
        [-1, 2, 2, -1, -1, -1, -1, -1, -1, -1],
        [-1, 2, 6, 5, -1, -1, -1, -1, -1, -1],
        [-1, 2, 7, 5, -1, -1, -1, -1, -1, -1]
    ]).T
    aqa = construct_aqa(job_table=job_table)
    aqa.weights = {1: 0.4, 2: 0.6}
    aqa.idle_status = -1
    assert aqa.adjust_weights(current_time=3) == {1: 0, 2: 1.0}