from peacsim.parsing import cluster_profile_reader
from io import StringIO

file = StringIO("[dell-1] "
                "\n; Idle power can be obtained with `Client.py --mode getIdle` "
                "\nidle_watts = 91.9 "
                "\n "
                "\n[dell-2] "
                "\nidle_watts = 91.8 "
                "\n "
                "\n[dell-3] "
                "\nidle_watts = 90 "
                )

cluster_config_reader = cluster_profile_reader.ClusterProfileReader(file)


def test_node_count():
    assert cluster_config_reader.node_count == 3


def test_client_ids_by_name():
    assert cluster_config_reader.node_ids == ['dell-1', 'dell-2', 'dell-3']


def test_all_idle_power():
    assert cluster_config_reader.all_idle_watts == {'dell-1': 91.9, 'dell-2': 91.8, 'dell-3': 90}
