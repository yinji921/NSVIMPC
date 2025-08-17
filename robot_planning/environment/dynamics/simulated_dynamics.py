import jax.numpy as np
import torch


class SimulatedDynamics(object):
    def __init__(self, dynamics_type=None, data_type=None, delta_t=None):
        self.dynamics_type = dynamics_type
        self.data_type = data_type
        self.delta_t = delta_t

    def initialize_from_config(self, config_data, section_name):
        self.dynamics_type = config_data.get(section_name, "type")
        if config_data.has_option(section_name, "delta_t"):
            self.delta_t = float(config_data.get(section_name, "delta_t"))

    def get_dynamics_type(self):
        return self.dynamics_type

    def get_data_type(self):
        return self.data_type

    def get_delta_t(self):
        return self.delta_t

    def get_state_dim(self):
        raise NotImplementedError

    def get_action_dim(self):
        raise NotImplementedError

    def get_max_action(self):
        raise NotImplementedError

    def get_state_bounds(self):
        raise NotImplementedError

    def propagate(self, state, action, delta_t=None):
        raise NotImplementedError

    def get_thrust_force(self, state, action):
        raise NotImplementedError

class NumpySimulatedDynamics(SimulatedDynamics):
    def __init__(self, dynamics_type=None, data_type=None, delta_t=None):
        SimulatedDynamics.__init__(self, dynamics_type, data_type, delta_t)
        self.data_type = np.ndarray

    def initialize_from_config(self, config_data, section_name):
        SimulatedDynamics.initialize_from_config(self, config_data, section_name)

    def propagate(self, state, action, delta_t=None):
        raise NotImplementedError


class PytorchSimulatedDynamics(SimulatedDynamics):
    def __init__(self, dynamics_type=None, data_type=None, delta_t=None):
        SimulatedDynamics.__init__(self, dynamics_type, data_type, delta_t)
        self.data_type = torch.tensor

    def initialize_from_config(self, config_data, section_name):
        SimulatedDynamics.initialize_from_config(self, config_data, section_name)

    def propagate(self, state, action, delta_t=None):
        raise NotImplementedError
