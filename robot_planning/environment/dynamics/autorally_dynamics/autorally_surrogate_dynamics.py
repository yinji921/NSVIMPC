import numpy as np
import torch
import scipy.sparse
import time
from math import fmod
import torch
from robot_planning.environment.dynamics.autorally_dynamics.autorally_dynamics import (
    AutoRallyDynamics,
)
from robot_planning.controllers.MPPI.utils.fit_inverse_throttle_nn import (
    load_inverse_throttle_model,
)
from robot_planning.environment.dynamics.autorally_dynamics import map_coords

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser


class ControlAffineAutoRallyDynamics(AutoRallyDynamics):
    """Wrap the AutoRally dynamics in control-affine form"""

    def __init__(self, dynamics_type=None, data_type=None, delta_t=None):
        # Refer back to the superclass initializer
        AutoRallyDynamics.__init__(
            self, "autorally_dynamics_cartesian", data_type, delta_t
        )
        self.inverse_throttle_model = load_inverse_throttle_model()

    def initialize_from_config(self, config_data, section_name):
        # Refer to superclass config loader
        AutoRallyDynamics.initialize_from_config(self, config_data, section_name)

        # load additional parameters
        self.v_ref = config_data.getfloat(section_name, "v_ref")
        self.steering_factor = config_data.getfloat(section_name, "steering_factor")

    def propagate(self, state, control, delta_t=None):
        state = state.copy().T
        control = control.copy().T

        if state.ndim == 1:
            state = state.reshape((1, -1))
            output_flat = True
        else:
            output_flat = False
        if control.ndim == 1:
            control = control.reshape((1, -1))

        # -------------------------------------------------------------------------------
        # Update the surrogate states, delta (1st dimension) and s (2nd dimension),
        # which represent steering angle and reference distance along the path
        # -------------------------------------------------------------------------------
        if delta_t is None:
            dt = self.get_delta_t()
        else:
            dt = delta_t
        # update steering angle based on command
        state[:, 0] += dt * self.steering_factor * control[:, 0]
        # update distance along path based on reference velocity
        state[:, 1] += dt * self.v_ref

        # -------------------------------------------------------------------------------
        # Convert surrogate state and control to autorally control
        # The autorally control is the steering angle and throttle command. The steering
        # angle is the 9th surrogate state dimension, and the throttle needs to be
        # found using the inverse throttle model to convert from rear wheel angular
        # acceleration (the second surrogate control input) to throttle.
        # -------------------------------------------------------------------------------

        # Create somewhere to store the autorally control input
        autorally_control = np.zeros_like(control)

        # Copy in the steering angle
        autorally_control[:, 0] = state[:, 0]

        # Get the throttle command using the fitted inverse model
        r_wheel_acceleration = control[:, 1]
        r_wheel_speed = state[:, 6]
        with torch.no_grad():
            throttle = self.inverse_throttle_model(
                torch.from_numpy(r_wheel_speed).reshape(-1, 1).float(),
                torch.from_numpy(r_wheel_acceleration).reshape(-1, 1).float(),
            )
        autorally_control[:, 1] = throttle.squeeze()
        # autorally_control[:, 1] = r_wheel_acceleration

        # -------------------------------------------------------------------------------
        # Convert surrogate state to autorally state
        # The cartesian autorally state is:
        #   [vx, vy, wz, wF, wR, psi, X, Y]
        # The surrogate state is:
        #   [delta, s_cumulative, vx, vy, wz, wF, wR, psi, X, Y]
        # -------------------------------------------------------------------------------

        # Copy in the relevant states
        autorally_state = state[:, 2:].copy()

        # -------------------------------------------------------------------------------
        # Propagate
        # -------------------------------------------------------------------------------
        next_state = AutoRallyDynamics.propagate(
            self, autorally_state.T, autorally_control.T, delta_t
        ).T

        # -------------------------------------------------------------------------------
        # Convert back to the CLBF state
        # -------------------------------------------------------------------------------
        state[:, 2:] = next_state

        # -------------------------------------------------------------------------------
        # Return
        # -------------------------------------------------------------------------------
        if output_flat:
            state = state.flatten()

        return state.T

    def surrogate_state_to_clbf_state(self, state):
        # Convert from surrogate state
        #   [delta, s_cumulative, vx, vy, wz, wF, wR, psi, X, Y]
        # to CLBF state
        #   [X_e, Y_e, psi_e, delta, wF_e, wR_e, vx, vy, psi_e_dot]

        # Create somewhere to store the results
        clbf_state = np.zeros_like(state[1:, :])

        SXE = 0
        SYE = 1
        PSI_E = 2
        DELTA = 3
        OMEGA_F_E = 4
        OMEGA_R_E = 5
        VX = 6
        VY = 7
        PSI_E_DOT = 8

        # Copy in the states that don't change
        clbf_state[DELTA, :] = state[0, :]  # delta
        clbf_state[VX, :] = state[2, :]
        clbf_state[VY, :] = state[3, :]

        # wF_e and wR_e are wheel speeds relative to a reference speed
        wF = state[5, :]
        wR = state[6, :]
        wF_e = wF - self.v_ref / self.rF
        wR_e = wR - self.v_ref / self.rR
        clbf_state[OMEGA_F_E, :] = wF_e
        clbf_state[OMEGA_R_E, :] = wR_e

        # Localize the cartesian state on the map to get the relative error
        psi_e, y_e, x_e, psi_ref_dot = self.track.localize(
            np.array((state[-2, :], state[-1, :])),
            state[-3, :],
            relative_to_s=state[1, :],
            return_ref_yaw_rate=True,
        )

        # Save to the CLBF state
        clbf_state[PSI_E_DOT, :] = state[4, :] - psi_ref_dot
        clbf_state[PSI_E, :] = psi_e
        clbf_state[SXE, :] = x_e
        clbf_state[SYE, :] = y_e

        return clbf_state

    def get_state_dim(self):
        return (10,)

    def get_action_dim(self):
        return (2,)

    def get_max_action(self):
        return np.array([10, 10])

    def torch_control_affine_dynamics(
        self,
        x: torch.Tensor,
        params,
    ):
        """
        Return a tuple (f, g) representing the system dynamics in control-affine form:

            dx/dt = f(x) + g(x) u

        args:
            x: bs x 9 tensor of state (from surrogate_state_to_clbf_state)
            params: a dictionary giving the parameter values for the system.
        returns:
            f: bs x self.get_state_dim() x 1 tensor representing the drift dynamics
            g: bs x self.get_state_dim() x self.n_controls tensor representing the
                control-dependent dynamics
        """
        # Sanity check on input
        assert x.ndim == 2
        assert x.shape[1] == 9

        return self._f(x, params), self._g(x, params)

    def _f(self, x: torch.Tensor, params):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x 9 tensor of state (from surrogate_state_to_clbf_state)
            params: a dictionary giving the parameter values for the system.
        returns:
            f: bs x 9 x 1 tensor
        """
        # State indices for CLBF state
        N_DIMS = 9
        SXE = 0
        SYE = 1
        PSI_E = 2
        DELTA = 3
        OMEGA_F_E = 4
        OMEGA_R_E = 5
        VX = 6
        VY = 7
        PSI_E_DOT = 8

        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, N_DIMS, 1))
        f = f.type_as(x)

        # Extract the parameters
        v_ref = torch.tensor(params["v_ref"])
        omega_ref = torch.tensor(params["omega_ref"])

        # set gravity constant
        g = 9.80665  # [m/s^2]

        # set parameters
        m_kg = self.mass
        Iz_kgm2 = self.Iz
        Iwf_kgm2 = self.IwF
        lf_m = self.lF
        lFR = self.lFR
        lr_m = lFR - lf_m
        rf_m = self.rR
        rr_m = self.rR
        h_m = self.height
        B = self.tire_B
        C = self.tire_C
        D = self.tire_D

        # Extract the state variables and adjust for the reference
        vx = x[:, VX]
        vy = x[:, VY]
        omega_f = x[:, OMEGA_F_E] + v_ref / rf_m
        omega_r = x[:, OMEGA_R_E] + v_ref / rr_m
        psi_e = x[:, PSI_E]
        psi_e_dot = x[:, PSI_E_DOT]
        psi_dot = psi_e_dot + omega_ref
        delta = x[:, DELTA]
        sxe = x[:, SXE]
        sye = x[:, SYE]

        # We want to express the error in x and y in the reference path frame, so
        # we need to get the dynamics of the rotated global frame error
        dsxe_r = vx * torch.cos(psi_e) - vy * torch.sin(psi_e) - v_ref + omega_ref * sye
        dsye_r = vx * torch.sin(psi_e) + vy * torch.cos(psi_e) - omega_ref * sxe
        f[:, SXE, 0] = dsxe_r
        f[:, SYE, 0] = dsye_r

        f[:, PSI_E, 0] = psi_e_dot  # integrate
        f[:, OMEGA_R_E, 0] = 0.0  # actuated
        f[:, DELTA, 0] = 0.0  # actuated

        # Compute front and rear coefficients of friction
        # This starts with wheel speeds
        v_fx = (
            vx * torch.cos(delta)
            + vy * torch.sin(delta)
            + psi_dot * lf_m * torch.sin(delta)
        )
        v_fy = (
            vy * torch.cos(delta)
            - vx * torch.sin(delta)
            + psi_dot * lf_m * torch.cos(delta)
        )
        v_rx = vx
        v_ry = vy - psi_dot * lr_m

        # From that, get longitudinal and lateral slip
        sigma_fx = (v_fx - omega_f * rf_m) / (1e-3 + omega_f * rf_m)
        sigma_fy = v_fy / (1e-3 + omega_f * rf_m)
        sigma_rx = (v_rx - omega_r * rr_m) / (1e-3 + omega_r * rr_m)
        sigma_ry = v_ry / (1e-3 + omega_r * rr_m)
        # And slip magnitude
        sigma_f = torch.sqrt(1e-5 + sigma_fx ** 2 + sigma_fy ** 2)
        sigma_r = torch.sqrt(1e-5 + sigma_rx ** 2 + sigma_ry ** 2)

        # These let us get friction coefficients
        mu_f = D * torch.sin(C * torch.arctan(B * sigma_f))
        mu_r = D * torch.sin(C * torch.arctan(B * sigma_r))

        # Decompose friction into longitudinal and lateral
        mu_fx = -sigma_fx / (sigma_f + 1e-3) * mu_f
        mu_fy = -sigma_fy / (sigma_f + 1e-3) * mu_f
        mu_rx = -sigma_rx / (sigma_r + 1e-3) * mu_r
        mu_ry = -sigma_ry / (sigma_r + 1e-3) * mu_r

        # Compute vertical forces on the front and rear wheels
        f_fz = (m_kg * g * lr_m - m_kg * g * mu_rx * h_m) / (
            lf_m
            + lr_m
            + mu_fx * h_m * torch.cos(delta)
            - mu_fy * h_m * torch.sin(delta)
            - mu_rx * h_m
        )
        f_rz = m_kg * g - f_fz

        # Get longitudinal and lateral wheel forces from vertical forces and friction
        f_fx = f_fz * mu_fx
        f_fy = f_fz * mu_fy
        f_rx = f_rz * mu_rx
        f_ry = f_rz * mu_ry

        # Use these to compute derivatives
        f[:, OMEGA_F_E, 0] = -rf_m / Iwf_kgm2 * f_fx

        f[:, PSI_E_DOT, 0] = (
            (f_fy * torch.cos(delta) + f_fx * torch.sin(delta)) * lf_m - f_ry * lr_m
        ) / Iz_kgm2 - omega_ref

        # Changes in vx and vy have three components. One due to the dynamics of the
        # car, one due to the rotation of the car,
        # and one due to the fact that the reference frame is rotating.

        # Dynamics
        vx_dot = (f_fx * torch.cos(delta) - f_fy * torch.sin(delta) + f_rx) / m_kg
        vy_dot = (f_fx * torch.sin(delta) + f_fy * torch.cos(delta) + f_ry) / m_kg

        # Car rotation and frame rotation
        vx_dot += vy * psi_e_dot
        vy_dot += -vx * psi_e_dot

        f[:, VX, 0] = vx_dot
        f[:, VY, 0] = vy_dot

        return f

    def _g(self, x: torch.Tensor, params):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x 9 tensor of state (from surrogate_state_to_clbf_state)
            params: a dictionary giving the parameter values for the system.
        returns:
            g: bs x 9 x self.n_controls tensor
        """
        # Extract batch size and set up a tensor for holding the result
        N_DIMS = 9
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, N_DIMS, self.get_action_dim()[0]))
        g = g.type_as(x)

        # Steering angle delta and rear wheel angular acceleration are controlled,
        # everything else is pure drift dynamics
        VDELTA = 0
        OMEGA_R_E_DOT = 1
        DELTA = 3
        OMEGA_R_E = 5
        g[:, DELTA, VDELTA] = 1.0
        g[:, OMEGA_R_E, OMEGA_R_E_DOT] = 1.0

        return g
