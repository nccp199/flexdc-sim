from tests.construct_aqa import construct_aqa

def aqa_setup():
    aqa = construct_aqa()
    aqa._idle_power = 10
    aqa._server_count = 10
    aqa.weighted_average_power = 50
    return aqa

def test_no_servers():
    aqa = aqa_setup()
    p_target = 100
    assert aqa.calculate_total_active_servers(p_target) == 0

def test_one_server():
    aqa = aqa_setup()
    p_target = 105
    assert aqa.calculate_total_active_servers(p_target) == 1

def test_multiple_servers():
    aqa = aqa_setup()
    p_target = 250
    assert aqa.calculate_total_active_servers(p_target) == 4

def test_all_servers():
    aqa = aqa_setup()
    p_target = 1000
    assert aqa.calculate_total_active_servers(p_target) == 10


