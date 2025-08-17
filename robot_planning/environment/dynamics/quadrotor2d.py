from functools import partial

import ipdb
import jax.numpy as np
import jax
import numpy
import scipy.sparse
import time
from math import fmod
import torch
from robot_planning.environment.dynamics.simulated_dynamics import (
    NumpySimulatedDynamics,
)
from robot_planning.factory.factories import kinematics_factory_base
import ast
from robot_planning.environment.dynamics.autorally_dynamics import throttle_model
from robot_planning.environment.dynamics.autorally_dynamics import map_coords
from robot_planning.factory.factories import noise_sampler_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.helper.utils import AUTORALLY_DYNAMICS_DIR
import random
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser


class QuadrotorDynamics2D(NumpySimulatedDynamics):
    """
    [x, z, theta, vx, vz, w]
    """
    def __init__(self, dynamics_type=None, data_type=None, delta_t=None):
        NumpySimulatedDynamics.__init__(self, dynamics_type, data_type, delta_t)
        # quadrotor parameters
        self.mass = 0.01 # (kg) mass
        self.l = 0.07  # (m) half the distance between the rotors
        self.I = (1/6)*self.mass* (np.sqrt(2)*self.l) # moment of inertia is (1/6)*m*a, where a is the side length of a square plane
        self.rho = 1.0 # empirical ground effect coefficient
        self.R = 0.02 # (m) length of the rotor propeller
        self.kinematics = None
        self.thrust_limits = [-5.0, 5.0]

    def initialize_from_config(self, config_data, section_name):
        NumpySimulatedDynamics.initialize_from_config(self, config_data, section_name)
        self.mass = config_data.getfloat(section_name, 'mass')
        self.I = config_data.getfloat(section_name, 'I')
        self.rho = config_data.getfloat(section_name, 'rho')
        kinematics_section_name = config_data.get(section_name, "kinematics")
        self.kinematics = factory_from_config(
            kinematics_factory_base, config_data, kinematics_section_name
        )
        self.l = self.kinematics.l
        self.R = self.kinematics.R

    @partial(jax.jit, static_argnums=(0,))
    def propagate(self, state, action, delta_t=None):
        print(f"re-tracing propagate!")

        delta_T = 0.01

        dt = self.get_delta_t()
        num_steps = dt // delta_T
        # ipdb.set_trace()
        if action.ndim == 1:
            action = action[:, None]
        if state.ndim == 1:
            state = state[:, None]

        # Input thrust of motor 1
        F1_in = action[0, :]
        # Input thrust by motor 2
        F2_in = action[1, :]

        def update_state_substep(state, dummy_input):
            single_state = False
            if state.ndim == 1:
                single_state = True
            if single_state:
                x = state[0]
                z = state[1]
                theta = state[2]
                vx = state[3]
                vz = state[4]
                w = state[5]
            else:
                x = state[0, :]
                z = state[1, :]
                theta = state[2, :]
                vx = state[3, :]
                vz = state[4, :]
                w = state[5, :]

            ## Compute the acceleration
            # Gravity constant
            g = 9.81
            # Actual Height of motor 1
            z1 = z - self.l * np.sin(theta)
            # Actual Height of motor 2
            z2 = z + self.l * np.sin(theta)

            coef_max = 10.0
            denom_min = 1 / coef_max
            denom_1 = np.maximum(denom_min, 1 - self.rho * (self.R / (4 * z1)) ** 2)
            denom_2 = np.maximum(denom_min, 1 - self.rho * (self.R / (4 * z2)) ** 2)

            # Actual thrust of motor 1
            F1 = F1_in / denom_1
            # Actual thrust of motor 2
            F2 = F2_in / denom_2

            # Total thrust
            F = F1 + F2
            F = np.clip(F, self.thrust_limits[0], self.thrust_limits[1])
            # Torque
            tau = self.l * (F2 - F1)

            x_dd = -F * np.sin(theta) / self.mass
            z_dd = F * np.cos(theta) / self.mass - g
            theta_dd = tau / self.I

            next_state = np.zeros_like(state)
            x = x + vx * delta_T
            z = z + vz * delta_T
            theta = theta + w * delta_T
            vx = vx + x_dd * delta_T
            vz = vz + z_dd * delta_T
            w = w + theta_dd * delta_T
            # jax.debug.print("vertical acceleration: {z_dd}", z_dd = z_dd)
            # jax.debug.print("x: {x}, z: {z}, theta: {theta}, vx: {vx}, vz: {vz}, w: {w}", x=x, z=z, theta=theta, vx=vx, vz=vz, w=w)
            next_state = next_state.at[0, :].set(x)
            next_state = next_state.at[1, :].set(z)
            next_state = next_state.at[2, :].set(theta)
            next_state = next_state.at[3, :].set(vx)
            next_state = next_state.at[4, :].set(vz)
            next_state = next_state.at[5, :].set(w)
            return next_state, None

        next_state, _ = jax.lax.scan(update_state_substep, state, xs=None, length=num_steps)
        next_state = next_state.squeeze()
        return next_state

    def get_state_dim(self):
        return (6,)

    def get_action_dim(self):
        return (2,)




