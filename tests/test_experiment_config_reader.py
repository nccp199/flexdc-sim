from peacsim.parsing import experiment_config_reader
from io import StringIO

file = StringIO("[system]"
                "\n;Standard deviation of performance variation distribution"
                "\nperformance_variation = 0.1"
                "\n;Control step seconds of the simulator"
                "\ntime_granularity = 10"
                "\n;random_seed for randomness"
                "\nrandom_seed = 120"
                "\n;Length of the simulation in minutes"
                "\nsimulation_duration = 60"
                "\n;Expected utilization ratio (or target utilization for simulation runs)"
                "\nutilization = 0.50"
                "\nidle_watts = 169"
                "\nserver_count = 1000"
                "\n[iso]"
                "\n;Boolean to normalize iso signal"
                "\nnormalize_iso_signal = True"
                "\n;ISO signal granularity in seconds"
                "\niso_signal_granularity = 4"
                "\n;file path for ISO24h.csv"
                "\niso_file_path = ../../data/ISO24h.csv"
                "\n;Boolean to randomize ISO start"
                "\nrandomize_iso_start = False"
                )
exp_reader = experiment_config_reader.ExperimentConfigReader(file)


def test_server_count():
    assert exp_reader.server_count == 1000


def test_simulation_performance_variation():
    assert exp_reader.performance_variation == 0.1


def test_time_granularity():
    assert exp_reader.time_granularity == 10


def test_random_seed():
    assert exp_reader.random_seed == 120


def test_simulation_duration():
    assert exp_reader.simulation_duration == 60


def test_utilization():
    assert exp_reader.utilization == 0.50


def test_normalize_iso_signal():
    assert exp_reader.normalize_iso_signal == True


def test_iso_signal_granularity():
    assert exp_reader.iso_signal_granularity == 4


def test_iso_file_path():
    assert exp_reader.iso_file_path == "../../data/ISO24h.csv"


def test_idle_power():
    assert exp_reader.idle_power == 169
