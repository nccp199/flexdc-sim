from peacsim.parsing import parsing_utils as parsing_utils
import configparser

class PolicyConfigReader:
    def __init__(self, file_source):
        """
        Initializes the PolicyConfigReader by reading from a configuration file.
        """
        self.policy_config = parsing_utils.init_config_parser(file_source)
        self._runtime_policy_name = self.policy_config.get("policy", "name")
        self._P = self.policy_config.getfloat("bid", "P")
        self._R = self.policy_config.getfloat("bid", "R")
        self._P_ratio = self.policy_config.getfloat("bid", "P_ratio")
        self._R_ratio = self.policy_config.getfloat("bid", "R_ratio")
        self._policy_parameters = {key: value for key, value in self.policy_config.items("parameters")}

        node_count_value = self._policy_parameters.get('node_count_control', 'true')
        self._policy_parameters['node_count_control'] = node_count_value.lower() == "true"

    @classmethod
    def from_parameters(cls, policy_name, P, R, P_ratio, R_ratio, policy_parameters):
        """
        Alternative constructor that initializes the PolicyConfigReader with explicit values.
        This constructor does not provide any defaults; you must supply all parameters.
        """
        instance = cls.__new__(cls)  # Create a new, uninitialized instance.
        instance._runtime_policy_name = policy_name
        instance._P = P
        instance._R = R
        instance._P_ratio = P_ratio
        instance._R_ratio = R_ratio
        instance._policy_parameters = policy_parameters
        return instance

    @property
    def P_ratio(self):
        return self._P_ratio

    @property
    def R_ratio(self):
        return self._R_ratio

    @property
    def runtime_policy_name(self):
        return self._runtime_policy_name

    @property
    def P(self):
        return self._P

    @property
    def R(self):
        return self._R

    @property
    def policy_parameters(self):
        return self._policy_parameters

    def to_dict(self):
        return {
            "runtime_policy_name": self._runtime_policy_name,
            "P": self._P,
            "R": self._R,
            "P_ratio": self._P_ratio,
            "R_ratio": self._R_ratio,
            "policy_parameters": self._policy_parameters
        }

    def write_policy_ini(self, policy_name, policy_parameters, file_path="optimal_policy.ini"):
        config = configparser.ConfigParser()

        config["policy"] = {"name": policy_name}
        config["bid"] = {
            "P": str(self._P),
            "R": str(self._R),
            "P_ratio": str(self._P_ratio),
            "R_ratio": str(self._R_ratio)
        }
        config["parameters"] = policy_parameters

        with open(file_path, "w") as configfile:
            config.write(configfile)
