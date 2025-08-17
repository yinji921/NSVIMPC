from functools import partial
import jax.debug as jd

import ipdb
import jax.numpy as np
import jax
import loguru
import numpy
import scipy.sparse
import time
from math import fmod
import torch
from robot_planning.environment.dynamics.simulated_dynamics import (
    NumpySimulatedDynamics,
)
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


class AutoRallyDynamics(NumpySimulatedDynamics):
    def __init__(self, dynamics_type=None, data_type=None, delta_t=None):
        NumpySimulatedDynamics.__init__(self, dynamics_type, data_type, delta_t)
        self.mass = 0.0
        self.Iz = 0.0
        self.lF = 0.0
        self.lFR = 0.0
        self.IwF = 0.0
        self.IwR = 0.0
        self.rF = 0.0
        self.rR = 0.0
        self.height = 0.0
        self.tire_B = 0.0
        self.tire_C = 0.0
        self.tire_D = 0.0
        self.kSteering = 0.0
        self.cSteering = 0.0
        self.throttle_factor = 0.0
        self.friction_nn = None
        self.throttle_nn = None
        self.track = None

        # decoded nn
        self.L1W = np.array(
            [
                [-4.0160503387e01, -1.0396156460e-01],
                [5.6865772247e01, 1.1424105167e00],
                [-2.7877355576e01, 8.1013500690e-02],
                [-2.7034116745e01, -2.0072770119e00],
                [3.0620580673e01, -1.6317946836e-02],
                [-3.4583007812e01, 2.2306546569e-02],
                [1.7588891983e01, -1.1739752442e-01],
                [-4.2060924530e01, 4.9316668883e-03],
                [2.8054933548e00, 6.7983172834e-02],
                [-2.0927755356e01, -2.7351467609e00],
                [-3.4222785950e01, -8.8916033506e-02],
                [-5.0679687500e01, -4.3296713829e00],
                [-3.8273872375e01, 6.3557848334e-02],
                [-6.8535380363e00, 1.0541532934e-01],
                [-5.7073421478e01, -1.8881739378e00],
                [3.9726573944e01, 5.3338561207e-02],
                [-3.1062200546e01, -3.2722020149e00],
                [3.2069881439e01, 2.1122207642e02],
                [-1.4632539368e02, 5.0957229614e01],
                [-7.5698246956e00, -1.2893879414e-01],
            ]
        )
        self.L1B = np.array(
            [
                4.9647192955,
                -11.6490354538,
                0.9536626935,
                30.6506881714,
                -7.0774221420,
                0.9992623329,
                10.6967773438,
                9.4048290253,
                -17.5587844849,
                1.5835133791,
                3.5548210144,
                6.9317874908,
                5.8543529510,
                -17.1659870148,
                7.7465252876,
                -9.3858194351,
                34.2410469055,
                -7.3937258720,
                -204.8706054688,
                0.6013393998,
            ]
        )
        self.L2W = np.array(
            [
                [
                    6.1452589035,
                    1.0189874172,
                    -25.4556179047,
                    3.2313644886,
                    43.4599952698,
                    20.1596012115,
                    17.1203384399,
                    -22.8726711273,
                    -90.1767959595,
                    3.3784775734,
                    5.1697301865,
                    8.3276357651,
                    -28.8062953949,
                    -26.1210975647,
                    9.5909786224,
                    28.0447864532,
                    2.8735139370,
                    3.8586735725,
                    3.5681331158,
                    -8.9000215530,
                ]
            ]
        )
        self.L2B = np.array(5.8514204025)

        if dynamics_type == "autorally_dynamics_map":
            self.cartesian = False
        elif dynamics_type == "autorally_dynamics_cartesian":
            self.cartesian = True
        else:
            self.cartesian = True

    def throttle_model_decode_nn(self, T, wR):
        inputs = np.hstack(
            (T.reshape((-1, 1)), wR.reshape((-1, 1)) / self.throttle_factor)
        )
        val = (
            np.matmul(
                1 / (1 + np.exp(-(np.matmul(inputs, self.L1W.T) + self.L1B))),
                self.L2W.T,
            )
            + self.L2B
        )
        return val.flatten()

    def throttle_model_no_nn(self, T, wR):
        val = (
            np.arctan(-(T - 5 + (wR - 30.0) * 0.06) * 4.0) * 90 / np.pi
            + 57
            - 0.00625 * np.exp(-(-10.00000 + 33.33333 * (T - 0.20000)))
        )
        val[val < -50] = float(-50.0)
        return val.flatten()

    def initialize_from_config(self, config_data, section_name):
        NumpySimulatedDynamics.initialize_from_config(self, config_data, section_name)
        self.mass = config_data.getfloat(section_name, "m")
        self.Iz = config_data.getfloat(section_name, "Iz")
        self.lF = config_data.getfloat(section_name, "lF")
        self.lFR = config_data.getfloat(section_name, "lFR")
        self.IwF = config_data.getfloat(section_name, "IwF")
        self.IwR = config_data.getfloat(section_name, "IwR")
        self.rF = config_data.getfloat(section_name, "rF")
        self.rR = config_data.getfloat(section_name, "rR")
        self.height = config_data.getfloat(section_name, "h")
        self.tire_B = config_data.getfloat(section_name, "tire_B")
        self.tire_C = config_data.getfloat(section_name, "tire_C")
        self.tire_D = config_data.getfloat(section_name, "tire_D")
        self.kSteering = config_data.getfloat(section_name, "kSteering")
        self.cSteering = config_data.getfloat(section_name, "cSteering")
        self.throttle_factor = config_data.getfloat(section_name, "throttle_factor")
        if config_data.has_option(section_name, "noise_sampler"):
            noise_sampler_section_name = config_data.get(section_name, "noise_sampler")
            self.noise_sampler = factory_from_config(
                noise_sampler_factory_base, config_data, noise_sampler_section_name
            )
        if config_data.has_option(section_name, "noise_covariance"):
            self.noise_covariance = np.diag(np.asarray(
            ast.literal_eval(config_data.get(section_name, "noise_covariance"))))
        else:
            self.noise_covariance = None
        if config_data.has_option(section_name, "state_dependent_noise"):
            self.state_dependent_noise = config_data.getboolean(section_name, "state_dependent_noise")
            self.track_width_for_noise_covariance_change = config_data.getfloat(section_name, "track_width_for_noise_covariance_change")
            self.close_to_boundary_noise_covariance_multiplier = config_data.getfloat(section_name, "close_to_boundary_noise_covariance_multiplier")
        if config_data.has_option(section_name, "throttle_nn_file_name"):
            throttle_nn_file_name = config_data.get(
                section_name, "throttle_nn_file_name"
            )
        else:
            throttle_nn_file_name = False
        if throttle_nn_file_name:
            throttle_nn_file_path = AUTORALLY_DYNAMICS_DIR + "/" + throttle_nn_file_name
            self.throttle_nn_torch = throttle_model.Net()
            self.throttle_nn_torch.load_state_dict(torch.load(throttle_nn_file_path, weights_only=True))
            self.throttle_nn = throttle_model.JaxNet(self.throttle_nn_torch)

        track_file_name = config_data.get(section_name, "track_file_name")
        track_path = AUTORALLY_DYNAMICS_DIR + "/" + track_file_name
        self.track = map_coords.MapCA(track_path)

    @partial(jax.jit, static_argnums=(0,))
    def propagate(self, state, control):
        print(f"re-tracing propagate!")
        state = state.copy().T
        input = control.copy().T
        m_Vehicle_m = self.mass
        m_Vehicle_Iz = self.Iz
        m_Vehicle_lF = self.lF
        lFR = self.lFR
        m_Vehicle_lR = lFR - m_Vehicle_lF
        m_Vehicle_IwF = self.IwF
        m_Vehicle_IwR = self.IwR
        m_Vehicle_rF = self.rF
        m_Vehicle_rR = self.rR
        m_Vehicle_h = self.height
        m_g = 9.80665

        tire_B = self.tire_B
        tire_C = self.tire_C
        tire_D = self.tire_D

        m_Vehicle_kSteering = self.kSteering
        m_Vehicle_cSteering = self.cSteering
        throttle_factor = self.throttle_factor

        if state.ndim == 1:
            state = state.reshape((1, -1))
            output_flat = True
        else:
            output_flat = False
        if input.ndim == 1:
            input = input.reshape((1, -1))
        steering = input[:, 0]
        delta = m_Vehicle_kSteering * steering + m_Vehicle_cSteering
        T = np.maximum(input[:, 1], 0)

        deltaT = 0.01
        dt = self.get_delta_t()
        # compute number of steps needed
        num_steps = dt // deltaT

        # Loop using LAX control flow
        def update_state_substep(state, dummy_input):
            # Unpack state
            vx = state[:, 0]
            vy = state[:, 1]
            wz = state[:, 2]
            wF = state[:, 3]
            wR = state[:, 4]

            if self.cartesian:
                psi = state[:, 5]
                X = state[:, 6]
                Y = state[:, 7]
            else:
                e_psi = state[:, 5]
                e_y = state[:, 6]
                s = state[:, 7]

            beta = np.arctan2(vy, vx)

            V = np.sqrt(vx * vx + vy * vy)
            vFx = V * np.cos(beta - delta) + wz * m_Vehicle_lF * np.sin(delta)
            vFy = V * np.sin(beta - delta) + wz * m_Vehicle_lF * np.cos(delta)
            vRx = vx
            vRy = vy - wz * m_Vehicle_lR

            # FIXME not sure if totally right
            # avoid division by 0
            wF = np.where(wF == 0, 1, wF)
            wR = np.where(wR == 0, 1, wR)

            sFx = np.where(wF > 0, (vFx - wF * m_Vehicle_rF) / (wF * m_Vehicle_rF), 0)
            sRx = np.where(wR > 0, (vRx - wR * m_Vehicle_rR) / (wR * m_Vehicle_rR), 0)
            sFy = np.where(vFx > 0, vFy / (wF * m_Vehicle_rF), 0)
            sRy = np.where(vRx > 0, vRy / (wR * m_Vehicle_rR), 0)

            sF = np.sqrt(sFx * sFx + sFy * sFy) + 1e-2
            sR = np.sqrt(sRx * sRx + sRy * sRy) + 1e-2

            muF = tire_D * np.sin(tire_C * np.arctan(tire_B * sF))
            muR = tire_D * np.sin(tire_C * np.arctan(tire_B * sR))

            muFx = -sFx / sF * muF
            muFy = -sFy / sF * muF
            muRx = -sRx / sR * muR
            muRy = -sRy / sR * muR

            fFz = (
                m_Vehicle_m
                * m_g
                * (m_Vehicle_lR - m_Vehicle_h * muRx)
                / (
                    m_Vehicle_lF
                    + m_Vehicle_lR
                    + m_Vehicle_h * (muFx * np.cos(delta) - muFy * np.sin(delta) - muRx)
                )
            )
            # fFz = m_Vehicle_m * m_g * (m_Vehicle_lR / 0.57)
            fRz = m_Vehicle_m * m_g - fFz

            fFx = fFz * muFx
            fRx = fRz * muRx
            fFy = fFz * muFy
            fRy = fRz * muRy

            ax = (
                fFx * np.cos(delta) - fFy * np.sin(delta) + fRx
            ) / m_Vehicle_m + vy * wz

            next_state = np.zeros_like(state)
            if self.friction_nn:
                input_tensor = torch.from_numpy(
                    np.vstack((steering, vx, vy, wz, ax, wF, wR)).T
                ).float()
                # input_tensor = torch.from_numpy(input).float()
                forces = self.friction_nn(input_tensor).detach().numpy()
                fafy = forces[:, 0]
                fary = forces[:, 1]
                fafx = forces[0, 2]
                farx = forces[0, 3]

                next_state = next_state.at[:, 0].set(
                    vx + deltaT * ((fafx + farx) / m_Vehicle_m + vy * wz)
                )
                next_state = next_state.at[:, 1].set(
                    vy + deltaT * ((fafy + fary) / m_Vehicle_m - vx * wz)
                )
                next_state = next_state.at[:, 2].set(
                    wz
                    + deltaT
                    * ((fafy) * m_Vehicle_lF - fary * m_Vehicle_lR)
                    / m_Vehicle_Iz
                )
            else:
                next_state = next_state.at[:, 0].set(
                    vx
                    + deltaT
                    * (
                        (fFx * np.cos(delta) - fFy * np.sin(delta) + fRx) / m_Vehicle_m
                        + vy * wz
                    )
                )
                next_state = next_state.at[:, 1].set(
                    vy
                    + deltaT
                    * (
                        (fFx * np.sin(delta) + fFy * np.cos(delta) + fRy) / m_Vehicle_m
                        - vx * wz
                    )
                )
                next_state = next_state.at[:, 2].set(
                    wz
                    + deltaT
                    * (
                        (fFy * np.cos(delta) + fFx * np.sin(delta)) * m_Vehicle_lF
                        - fRy * m_Vehicle_lR
                    )
                    / m_Vehicle_Iz
                )
            next_state = next_state.at[:, 3].set(
                wF - deltaT * m_Vehicle_rF / m_Vehicle_IwF * fFx
            )
            if self.throttle_nn:
                input_array = np.hstack(
                    (T.reshape((-1, 1)), wR.reshape((-1, 1)) / throttle_factor)
                )
                next_state = next_state.at[:, 4].set(
                    wR
                    + deltaT * self.throttle_nn(input_array).flatten()
                )
            else:
                # next_state[:, 4] = T  # wR + deltaT * (m_Vehicle_kTorque * (T-wR) - m_Vehicle_rR * fRx) / m_Vehicle_IwR
                # see /fit_throttle_nn.py for detail
                # dt_wR = np.arctan(-(T -5 + (wR-30)*0.06)*4) * 90/np.pi + 57
                dt_wR = self.throttle_model_no_nn(T, wR)
                next_state = next_state.at[:, 4].set(wR + deltaT * dt_wR)

            if self.cartesian or cartesian:
                next_state = next_state.at[:, 5].set(psi + deltaT * wz)
                next_state = next_state.at[:, 6].set(
                    X + deltaT * (np.cos(psi) * vx - np.sin(psi) * vy)
                )
                next_state = next_state.at[:, 7].set(
                    Y + deltaT * (np.sin(psi) * vx + np.cos(psi) * vy)
                )
            else:
                # rho = np.zeros_like(s).flatten()
                rho = self.track.get_cur_reg_from_s(s)[4].flatten()
                next_state = next_state.at[:, 5].set(
                    e_psi
                    + deltaT
                    * (
                        wz
                        - (vx * np.cos(e_psi) - vy * np.sin(e_psi))
                        / (1 - rho * e_y)
                        * rho
                    )
                )
                next_state = next_state.at[:, 6].set(
                    e_y + deltaT * (vx * np.sin(e_psi) + vy * np.cos(e_psi))
                )
                next_state = next_state.at[:, 7].set(
                    s
                    + deltaT
                    * (vx * np.cos(e_psi) - vy * np.sin(e_psi))
                    / (1 - rho * e_y)
                )

            return next_state, None

        next_state, _ = jax.lax.scan(update_state_substep, state, xs=None, length=num_steps)



        if output_flat:
            next_state = next_state.flatten()

        # assert self.noise_covariance is None
        if self.noise_covariance is not None:
            mean = np.zeros(self.noise_covariance.shape[0])
            noise_covariance = self.noise_covariance
            seed = ((np.sum(state[0, :]) + np.sum(control))*1000).astype(int) # generate random seed
            key = jax.random.PRNGKey(seed)
            noises = jax.random.multivariate_normal(key,
                mean, noise_covariance, (next_state.shape[0],))
            if self.state_dependent_noise is True:
                local_state = self.global_to_local_coordinate_transform(state)
                close_to_boundary_indices_a = local_state[:, -2]< self.track_width_for_noise_covariance_change
                close_to_boundary_indices_b = local_state[:, -2] > -self.track_width_for_noise_covariance_change
                close_to_boundary_indices = np.logical_and(close_to_boundary_indices_a, close_to_boundary_indices_b)
                # jax.debug.print("ey: {}", local_state[:, -2])
                # jax.debug.print("close_to_boundary_indices: {}", close_to_boundary_indices)
                close_to_boundary_indices = np.repeat(close_to_boundary_indices[:, np.newaxis], noises.shape[1], axis=1)
                close_to_boundary_indices = np.where(close_to_boundary_indices, 1, np.sqrt(self.close_to_boundary_noise_covariance_multiplier))
                # Magnify the noises on the states that has e_y >= self.track_width_for_noise_covariance_change or e_y <= -self.track_width_for_noise_covariance_change
                noises = noises * close_to_boundary_indices
            # jax.debug.print(np.array_str(noises)) # This does not work, because noises is a traced object
            next_state += noises
        return next_state.T

    def get_state_dim(self):
        return (8,)

    def get_action_dim(self):
        return (2,)

    def get_max_action(self):
        return np.array([1, 1])

    def shutdown(self):
        return

    def global_to_local_coordinate_transform(self, state):
        state = state.T
        e_psi, e_y, s = self.track.localize(
            np.array((state[-2, :], state[-1, :])), state[-3, :]
        )
        new_state = state.copy()
        new_state = new_state.at[-3:, :].set(np.vstack((e_psi, e_y, s)))
        return new_state.T