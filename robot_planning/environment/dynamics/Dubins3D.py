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


class Dubins3D(NumpySimulatedDynamics):
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
        self.speed = 1.0
        self.action_max = 1.0
        self.delta_t = self.get_delta_t()

    @partial(jax.jit, static_argnums=(0,))
    def propagate(self, state, action, delta_t=None):
        print(f"re-tracing propagate!")

        if action.ndim == 1:
            action = action[:, None]
        if state.ndim == 1:
            state = state[:, None]

        x = state[0, :]
        y = state[1, :]
        theta = state[2, :]

        dxdt = self.speed*np.cos(theta)
        dydt = self.speed*np.sin(theta)
        dthetadt = action[0, :]

        x = x + dxdt * self.delta_t
        y = y + dydt * self.delta_t
        theta = theta + dthetadt * self.delta_t

        state_next = np.zeros_like(state)
        state_next = state_next.at[0, :].set(x)
        state_next = state_next.at[1, :].set(y)
        state_next = state_next.at[2, :].set(theta)

        state_next = state_next.squeeze()
        return state_next

    def get_state_dim(self):
        return (3,)

    def get_action_dim(self):
        return (1,)

    def get_max_action(self):
        return np.array([1,1])

    def get_state_bounds(self):
        return self.state_bounds

