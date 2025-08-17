import jax.numpy as np
from functools import partial
import jax
from robot_planning.environment.dynamics.simulated_dynamics import NumpySimulatedDynamics
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import cost_evaluator_factory_base
import ast
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser


class BicycleDynamics(NumpySimulatedDynamics):
    def __init__(self, dynamics_type=None, data_type=None, delta_t=None, mass=None, cog_pos=None, car_length=None, state_bounds=None):
        NumpySimulatedDynamics.__init__(self, dynamics_type, data_type, delta_t)
        self.dynamics_type = dynamics_type
        self.data_type = data_type
        self.delta_t = delta_t
        self.mass = mass
        self.cog_pos = cog_pos
        self.car_length = car_length
        self.state_bounds = state_bounds

    def initialize_from_config(self, config_data, section_name):
        NumpySimulatedDynamics.initialize_from_config(self, config_data, section_name)
        self.mass = config_data.getfloat(section_name, 'mass')
        self.cog_pos = config_data.getfloat(section_name, 'cog_pos')
        self.car_length = config_data.getfloat(section_name, 'car_length')
        if config_data.has_option(section_name, 'state_bounds'):
            self.state_bounds = np.asarray(ast.literal_eval(config_data.get(section_name, 'state_bounds')))
    @partial(jax.jit, static_argnums=(0,))
    def propagate(self, state, action, delta_t=None):
        print(f"re-tracing propagate!")
        single_state = False
        if state.ndim == 1:
            single_state = True
        if single_state:
            x = state[0]
            y = state[1]
            Phi = state[2]
            delta = state[3]
            v = state[4]
        else:
            x = state[0, :]
            y = state[1, :]
            Phi = state[2, :]
            delta = state[3, :]
            v = state[4, :]

        lr = np.dot(self.car_length, self.cog_pos)
        beta = np.arctan(lr * np.tan(delta) / self.car_length)

        dxdt = v * np.cos(beta + Phi)
        dydt = v * np.sin(beta + Phi)
        dPhidt = v * np.tan(delta) * np.cos(beta) / self.car_length
        ddeltadt = action[0]
        dvdt = action[1]/self.mass

        x = x + dxdt * self.delta_t
        y = y + dydt * self.delta_t
        Phi = Phi + dPhidt * self.delta_t
        delta = delta + ddeltadt * self.delta_t
        v = v + dvdt * self.delta_t
        if single_state:
            state_next = np.array([x, y, Phi, delta, v]).reshape((5,))
        else:
            state_next = np.array([x, y, Phi, delta, v]).reshape((5, -1))
        return state_next

    # def propagate(self, state, action, delta_t=None):
    #     if state.size != 5:
    #         raise ValueError("Wrong state size! The bicycle model state has a dimensionality of 5")
    #     if action.size != 2:
    #         raise ValueError("Wrong state size! The bicycle model input has a dimensionality of 2")
    #     x = state[0]
    #     y = state[1]
    #     Phi = state[2]
    #     delta = state[3]
    #     v = state[4]
    #     lr = np.dot(self.car_length, self.cog_pos)
    #     beta = np.arctan(lr * np.tan(delta) / self.car_length)
    #     dxdt = v * np.cos(beta + Phi)
    #     dydt = v * np.sin(beta + Phi)
    #     dPhidt = v * np.tan(delta) * np.cos(beta) / self.car_length
    #     ddeltadt = action[0]
    #     dvdt = action[1]/self.mass
    #     x = x + dxdt * self.delta_t
    #     y = y + dydt * self.delta_t
    #     Phi = Phi + dPhidt * self.delta_t
    #     delta = delta + ddeltadt * self.delta_t
    #     v = v + dvdt * self.delta_t
    #     state_next = np.array([x, y, Phi, delta, v]).reshape((5,))
    #     return state_next

    def get_state_dim(self):
        return (5,)

    def get_action_dim(self):
        return (2,)

    def get_max_action(self):
        return np.array([1,1])

    def get_state_bounds(self):
        return self.state_bounds

