from peacsim.parsing import parsing_utils as parsing_utils


class SimulatedAnnealingConfigReader:
    def __init__(self, file_source):
        self.sa_config = parsing_utils.init_config_parser(file_source)

        # Initialization parameters
        self._p_init = self.sa_config.getfloat('initialization', 'p_init')
        self._r_init = self.sa_config.getfloat('initialization', 'r_init')

        # Simulated Annealing parameters
        self._temperature = self.sa_config.getint('simulated_annealing', 'temperature')
        self._cooling_rate = self.sa_config.getfloat('simulated_annealing', 'cooling_rate')
        self._max_iter = self.sa_config.getint('simulated_annealing', 'max_iter')
        self._smart_weight = self.sa_config.getboolean('simulated_annealing', 'smart_weight')

        # Boundaries
        self._p_low = self.sa_config.getfloat('boundaries', 'p_low')
        self._p_high = self.sa_config.getfloat('boundaries', 'p_high')
        self._r_low = self.sa_config.getfloat('boundaries', 'r_low')
        self._r_high = self.sa_config.getfloat('boundaries', 'r_high')
        self._w_low = self.sa_config.getfloat('boundaries', 'w_low')
        self._w_high = self.sa_config.getfloat('boundaries', 'w_high')

        # Step Size
        self._r_step = self.sa_config.getfloat('step_size', 'r_step')
        self._p_step = self.sa_config.getfloat('step_size', 'p_step')
        self._w_step = self.sa_config.getfloat('step_size', 'w_step')

        # Cost Function
        self._psi1 = self.sa_config.getfloat('cost_function', 'psi1')
        self._psi2 = self.sa_config.getfloat('cost_function', 'psi2')
        self._tracking_error_constraint = self.sa_config.getfloat('cost_function', 'TRACKING_ERROR_CONSTRAINT')
        self._beta = self.sa_config.getfloat('cost_function', 'beta')
        self._rho = self.sa_config.getfloat('cost_function', 'rho')
        self._qos_constraint = self.sa_config.getfloat('cost_function', 'qos_constraint')

        # DR Program
        self._program_type = self.sa_config.get('dr_program', 'program_type')
        if self._program_type == 'EDR':
            self._p_base = self.sa_config.getfloat('dr_program', 'p_base')
            self._piI = self.sa_config.getfloat('dr_program', 'piI')

    @property
    def p_init(self):
        return self._p_init

    @property
    def r_init(self):
        return self._r_init

    @property
    def temperature(self):
        return self._temperature

    @property
    def cooling_rate(self):
        return self._cooling_rate

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def smart_weight(self):
        return self._smart_weight

    @property
    def p_low(self):
        return self._p_low

    @property
    def p_high(self):
        return self._p_high

    @property
    def r_low(self):
        return self._r_low

    @property
    def r_high(self):
        return self._r_high

    @property
    def w_low(self):
        return self._w_low

    @property
    def w_high(self):
        return self._w_high

    @property
    def r_step(self):
        return self._r_step

    @property
    def p_step(self):
        return self._p_step

    @property
    def w_step(self):
        return self._w_step

    @property
    def psi1(self):
        return self._psi1

    @property
    def psi2(self):
        return self._psi2

    @property
    def tracking_error_constraint(self):
        return self._tracking_error_constraint

    @property
    def beta(self):
        return self._beta

    @property
    def rho(self):
        return self._rho

    @property
    def qos_constraint(self):
        return self._qos_constraint

    @property
    def program_type(self):
        return self._program_type

    @property
    def p_base(self):
        return self._p_base

    @property
    def piI(self):
        return self._piI

    def to_dict(self):
        return {
            "p_init": self._p_init,
            "r_init": self._r_init,
            "temperature": self._temperature,
            "cooling_rate": self._cooling_rate,
            "max_iter": self._max_iter,
            "smart_weight": self._smart_weight,
            "p_low": self._p_low,
            "p_high": self._p_high,
            "r_low": self._r_low,
            "r_high": self._r_high,
            "w_low": self._w_low,
            "w_high": self._w_high,
            "r_step": self._r_step,
            "p_step": self._p_step,
            "w_step": self._w_step,
            "psi1": self._psi1,
            "psi2": self._psi2,
            "tracking_error_constraint": self._tracking_error_constraint,
            "beta": self._beta,
            "rho": self._rho,
            "qos_constraint": self._qos_constraint,
            "program_type": self._program_type
        }
