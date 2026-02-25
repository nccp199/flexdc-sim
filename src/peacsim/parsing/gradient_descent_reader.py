from peacsim.parsing import parsing_utils as parsing_utils


class GradientDescentReader:
    def __init__(self, file_source):
        self.gradient_descent_config = parsing_utils.init_config_parser(file_source)
        self._sections = self.gradient_descent_config.sections()

        general_variables = {}
        for variable in self.gradient_descent_config.options('general_variables'):
            if variable == 'suffix' or variable == 'opt_weights_path':
                general_variables[variable] = self.gradient_descent_config.get('general_variables', variable)
            else:
                general_variables[variable] = self.gradient_descent_config.getfloat('general_variables', variable)
        self._general_variables = general_variables

        run_simulator_variables = {variable: self.gradient_descent_config.getfloat('run_simulator', variable)
                                   for variable in self.gradient_descent_config.options('run_simulator')}
        self._run_simulator_variables = run_simulator_variables

        calculate_delay_prob_variables = {
            variable: self.gradient_descent_config.getfloat('calculate_delay_prob', variable)
            for variable in self.gradient_descent_config.options('calculate_delay_prob')}
        self._calculate_delay_prob_variables = calculate_delay_prob_variables

        calculate_gradient_variables = {variable: self.gradient_descent_config.getfloat('calculate_gradient', variable)
                                        for variable in self.gradient_descent_config.options('calculate_gradient')}
        self._calculate_gradient_variables = calculate_gradient_variables

        gradient_driver_variables = {variable: self.gradient_descent_config.getfloat('gradient_driver', variable)
                                     for variable in self.gradient_descent_config.options('gradient_driver')}
        self._gradient_driver_variables = gradient_driver_variables

        self._dr_program_variables = {variable: self.gradient_descent_config.get('dr_program', variable)
                                      for variable in self.gradient_descent_config.options('dr_program')}

        self._use_lagrange = False
        self.lagrange_variables = {}
        if self.gradient_descent_config.has_section('lagrange'):
            self._use_lagrange = True
            self._qos_lagrange = self.gradient_descent_config.getfloat('lagrange', 'qos_lagrange')
            self._tracking_lagrange = self.gradient_descent_config.getfloat('lagrange', 'tracking_lagrange')
            self._qos_lagrange_step_size = self.gradient_descent_config.getfloat('lagrange', 'qos_lagrange_step_size')
            self._tracking_lagrange_step_size = \
                self.gradient_descent_config.getfloat('lagrange', 'tracking_lagrange_step_size')
            self.lagrange_variables = {
                'qos_lagrange': self._qos_lagrange,
                'tracking_lagrange': self._tracking_lagrange,
                'qos_lagrange_step_size': self._qos_lagrange_step_size,
                'tracking_lagrange_step_size': self._tracking_lagrange_step_size
            }

        self._use_augmented_lagrange = False
        self.augmented_lagrange_variables = {}
        self._c_k = None
        self._c_t = None
        self._c_k_rate = None
        self._c_t_rate = None
        if self.gradient_descent_config.has_section('augmented_lagrange'):
            self._use_augmented_lagrange = True
            self._c_k = self.gradient_descent_config.getfloat('augmented_lagrange', 'c_k')
            self._c_t = self.gradient_descent_config.getfloat('augmented_lagrange', 'c_t')
            self._c_k_rate = self.gradient_descent_config.getfloat('augmented_lagrange', 'c_k_rate')
            self._c_t_rate = self.gradient_descent_config.getfloat('augmented_lagrange', 'c_t_rate')
            self.augmented_lagrange_variables = {
                'c_k': self._c_k,
                'c_t': self._c_t,
                'c_k_rate': self._c_k_rate,
                'c_t_rate': self._c_t_rate
            }

        print("General variables:", self._general_variables)
        print("calculate_delay_prob_variables:", self._calculate_delay_prob_variables)
        print("calculate_gradient_variables:", self._calculate_gradient_variables)
        print("GD_variables:", self._gradient_driver_variables)

    @property
    def sections(self):
        return self._sections

    @property
    def general_variables(self):
        return self._general_variables

    @property
    def run_simulator_variables(self):
        return self._run_simulator_variables

    @property
    def calculate_delay_prob_variables(self):
        return self._calculate_delay_prob_variables

    @property
    def calculate_gradient_variables(self):
        return self._calculate_gradient_variables

    @property
    def gradient_driver_variables(self):
        return self._gradient_driver_variables

    @property
    def dr_program_variables(self):
        return self._dr_program_variables

    @property
    def use_lagrange(self):
        return self._use_lagrange

    @property
    def qos_lagrange(self):
        return self._qos_lagrange

    @property
    def tracking_lagrange(self):
        return self._tracking_lagrange

    @property
    def qos_lagrange_step_size(self):
        return self._qos_lagrange_step_size

    @property
    def tracking_lagrange_step_size(self):
        return self._tracking_lagrange_step_size

    @property
    def use_augmented_lagrange(self):
        return self._use_augmented_lagrange

    @property
    def c_k(self):
        return self._c_k

    @property
    def c_t(self):
        return self._c_t

    @property
    def c_k_rate(self):
        return self._c_k_rate

    @property
    def c_t_rate(self):
        return self._c_t_rate

    def to_dict(self):
        """Serialize all configuration sections into a dictionary."""
        return {
            "general_variables": self._general_variables,
            "run_simulator_variables": self._run_simulator_variables,
            "calculate_delay_prob_variables": self._calculate_delay_prob_variables,
            "calculate_gradient_variables": self._calculate_gradient_variables,
            "gradient_driver_variables": self._gradient_driver_variables,
            "lagrange_variables": self.lagrange_variables,
            "augmented_lagrange_variables": self.augmented_lagrange_variables,
            "dr_program_variables": self._dr_program_variables,
        }
