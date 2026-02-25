from peacsim.parsing import parsing_utils as parsing_utils
import configparser


class ExperimentConfigReader:
    def __init__(self, file_source):
        self.experiment_config = parsing_utils.init_config_parser(file_source)
        self._performance_variation = self.experiment_config.getfloat('system', 'performance_variation',
                                                                      fallback=None)
        self._time_granularity = self.experiment_config.getint('system', 'time_granularity', fallback=1)
        self._random_seed = self.experiment_config.getint('system', 'random_seed')
        self._simulation_duration = self.experiment_config.getint('system', 'simulation_duration')
        self._utilization = self.experiment_config.getfloat('system', 'utilization')
        self._normalize_iso_signal = self.experiment_config.getboolean('iso', 'normalize_iso_signal')
        self._randomize_iso_start = self.experiment_config.getboolean('iso', 'randomize_iso_start')
        self._iso_signal_granularity = self.experiment_config.getint('iso', 'iso_signal_granularity')
        self._iso_file_path = self.experiment_config.get('iso', 'iso_file_path')
        self._server_count = self.experiment_config.getint('system', 'server_count')
        self._idle_power = self.experiment_config.getfloat('system', 'idle_watts')
        self._workload_trace = self.experiment_config.get('system', 'workload_trace', fallback='poisson')
        try:
            self._iso_signal_start_hour = self.experiment_config.getint('iso', 'iso_signal_start_hour', fallback=None)
        except:
            self._iso_signal_start_hour = None

        try:
            spike_rate_str = self.experiment_config.get('system', 'spike_rate', fallback=None)
            self._spike_rate = float(spike_rate_str) if spike_rate_str and spike_rate_str.lower() != 'none' else None
        except (configparser.NoOptionError, configparser.NoSectionError):
            self._spike_rate = None

    @property
    def random_seed(self):
        return self._random_seed

    @property
    def performance_variation(self):
        return self._performance_variation

    @property
    def simulation_duration(self):
        return self._simulation_duration

    @property
    def utilization(self):
        return self._utilization

    @property
    def time_granularity(self):
        return self._time_granularity

    @property
    def normalize_iso_signal(self):
        return self._normalize_iso_signal

    @property
    def randomize_iso_start(self):
        return self._randomize_iso_start

    @property
    def iso_signal_granularity(self):
        return self._iso_signal_granularity

    @property
    def iso_file_path(self):
        return self._iso_file_path

    @property
    def iso_signal_start_hour(self):
        return self._iso_signal_start_hour

    @property
    def idle_power(self):
        return self._idle_power

    @property
    def server_count(self):
        return self._server_count

    @property
    def workload_trace(self):
        return self._workload_trace

    @property
    def spike_rate(self):
        return self._spike_rate

    def to_dict(self):
        return {
            "performance_variation": self._performance_variation,
            "time_granularity": self._time_granularity,
            "random_seed": self._random_seed,
            "simulation_duration": self._simulation_duration,
            "utilization": self._utilization,
            "normalize_iso_signal": self._normalize_iso_signal,
            "iso_signal_granularity": self._iso_signal_granularity,
            "iso_file_path": self._iso_file_path,
            "iso_signal_start_hour": self._iso_signal_start_hour,
            "server_count": self._server_count,
            "idle_power": self._idle_power,
            "workload_trace": self._workload_trace,
            "spike_rate": self._spike_rate,
        }

    def write_experiment_config(self, file_path="experiment_config.ini"):
        config = configparser.ConfigParser()

        config["system"] = {
            # "performance_variation": str(self._performance_variation),
            "time_granularity": str(self._time_granularity),
            "random_seed": str(self._random_seed),
            "simulation_duration": str(self._simulation_duration),
            "utilization": str(self._utilization),
            "server_count": str(self._server_count),
            "idle_watts": str(self._idle_power),
            "workload_trace": str(self._workload_trace),
            "spike_rate": str(self._spike_rate),
            "iso_signal_start_hour": str(self._iso_signal_start_hour)
        }

        config["iso"] = {
            "normalize_iso_signal": str(self._normalize_iso_signal),
            "randomize_iso_start": str(self._randomize_iso_start),
            "iso_signal_granularity": str(self._iso_signal_granularity),
            "iso_file_path": self._iso_file_path,
            "iso_signal_start_hour": str(self._iso_signal_start_hour)
        }

        with open(file_path, "w") as configfile:
            config.write(configfile)
