from peacsim.parsing import parsing_utils as parsing_utils
import configparser

class JobProfileReader:

    def __init__(self, file_source, scale_workload=1, force_qos=-1,
                 multi_hour=1):
        self._workload_mix_name = file_source
        self.job_profile_config = parsing_utils.init_config_parser(file_source)

        self._all_jobs = {job_index: job_name for (
            job_index, job_name) in enumerate(self.job_profile_config.sections())}
        self._scale_workload = scale_workload
        self._force_qos = force_qos
        self._multi_hour = multi_hour

        self._all_min_execution_time = {
            job_index: self.job_profile_config.getfloat(job_name, 'min_time_seconds') / self._scale_workload
            for (job_index, job_name)
            in self._all_jobs.items()}
        self._all_max_execution_time = {
            job_index: self.job_profile_config.getfloat(job_name, 'max_time_seconds') / self._scale_workload
            for (job_index, job_name)
            in self._all_jobs.items()}

        self._all_job_power_min = {job_index: self.job_profile_config.getfloat(job_name, 'min_job_power_watts')
                                   for (job_index, job_name)
                                   in self._all_jobs.items()}
        self._all_job_power_max = {job_index: self.job_profile_config.getfloat(job_name, 'max_job_power_watts')
                                   for (job_index, job_name)
                                   in self._all_jobs.items()}

        self._all_job_qos_constraints = {job_index: self.job_profile_config.getfloat(job_name, 'qos_constraint')
                                         for (job_index, job_name) in
                                         self._all_jobs.items()}
        # TODO: Find a more intuitive variable instead of multi_hour
        # TODO: Why there is a scaleWorkload
        if self._multi_hour:
            self._all_job_arrival_rate = {
                job_index: self.job_profile_config.getfloat(job_name, 'arrival_rate', fallback=1) * self._scale_workload
                for (job_index, job_name)
                in self._all_jobs.items()}

        self._all_job_size = {job_index: self.job_profile_config.getint(job_name, 'job_size', fallback=1)
                              for (job_index, job_name) in
                              self._all_jobs.items()}
        self._job_type_count = len(self._all_jobs)

    @property
    def job_type_count(self):
        return self._job_type_count

    @property
    def all_jobs(self):
        """Return all jobs"""
        return self._all_jobs
    
    @property
    def all_min_execution_time(self):
        """Return the min execution time"""
        return self._all_min_execution_time

    @property
    def all_max_execution_time(self):
        """Return the max execution time"""
        return self._all_max_execution_time

    @property
    def all_min_job_power(self):
        """Return the minimum job power"""
        return self._all_job_power_min

    @property
    def all_max_job_power(self):
        """Return the maximum job power"""
        return self._all_job_power_max

    @property
    def all_job_qos_constraints(self):
        """Return the job QoS constraints"""
        return self._all_job_qos_constraints

    @property
    def job_arrival_rate(self):
        """Return the job arrival rate"""
        return self._all_job_arrival_rate

    @property
    def all_job_size(self):
        """Return the job size"""
        return self._all_job_size

    @property
    def workload_mix_name(self):
        """Return the job size"""
        return self._workload_mix_name

    def to_dict(self):
        return {
            "all_jobs": self._all_jobs,
            "all_min_execution_time": self._all_min_execution_time,
            "all_max_execution_time": self._all_max_execution_time,
            "all_min_job_power": self._all_job_power_min,
            "all_max_job_power": self._all_job_power_max,
            "all_job_qos_constraints": self._all_job_qos_constraints,
            "all_job_arrival_rate": getattr(self, "_all_job_arrival_rate", {}),
            "all_job_size": self._all_job_size,
            "job_type_count": self._job_type_count
        }

    def write_job_profile_config(self, file_path="job_profile_config.ini"):
        config = configparser.ConfigParser()

        for job_index, job_name in self._all_jobs.items():
            config[job_name] = {
                "min_time_seconds": str(self._all_min_execution_time[job_index] * self._scale_workload),
                "max_time_seconds": str(self._all_max_execution_time[job_index] * self._scale_workload),
                "min_job_power_watts": str(self._all_job_power_min[job_index]),
                "max_job_power_watts": str(self._all_job_power_max[job_index]),
                "qos_constraint": str(self._all_job_qos_constraints[job_index]),
                "job_size": str(self._all_job_size[job_index])
            }

            if self._multi_hour:
                config[job_name]["arrival_rate"] = str(
                    self._all_job_arrival_rate.get(job_index, 1) / self._scale_workload)

        with open(file_path, "w") as configfile:
            config.write(configfile)
