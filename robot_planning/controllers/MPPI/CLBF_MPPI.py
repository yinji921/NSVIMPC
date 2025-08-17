"""Wraps an MPPI controller with a CLBF certificate for safety/stability"""
from robot_planning.controllers.MPPI.MPPI import MPPI
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factories import cost_evaluator_factory_base
from robot_planning.factory.factories import (
    stochastic_trajectories_sampler_factory_base,
)
import numpy as np
import torch
import torch.nn.functional as F
import gurobipy as gp
from gurobipy import GRB
import ast
import copy


class CLBF_MPPI(MPPI):
    def __init__(
        self,
        control_horizon=None,
        dynamics=None,
        cost_evaluator=None,
        control_dim=None,
        inverse_temperature=None,
        initial_control_sequence=None,
        stochastic_trajectories_sampler=None,
        renderer=None,
        clbf_lambda=None,
        clbf_relaxation_penalty=None,
    ):
        # Call superclass initialization
        MPPI.__init__(
            self,
            control_horizon,
            dynamics,
            cost_evaluator,
            control_dim,
            inverse_temperature,
            initial_control_sequence,
            stochastic_trajectories_sampler,
            renderer,
        )

        # Save CLBF-specific parameters
        clbf_lambda = clbf_lambda
        clbf_relaxation_penalty = clbf_relaxation_penalty

        self.scenarios = []

    def initialize_from_config(self, config_data, section_name):
        # Pass up to MPPI superclass
        MPPI.initialize_from_config(self, config_data, section_name)

        # Get CLBF-specific parameters
        self.clbf_lambda = config_data.getfloat(section_name, "clbf_lambda")
        self.clbf_relaxation_penalty = config_data.getfloat(
            section_name, "clbf_relaxation_penalty"
        )

        self.nominal_params = {
            "v_ref": config_data.getfloat(section_name, "v_ref"),
            "omega_ref": 0.0,
        }
        omega_max = config_data.getfloat(section_name, "omega_max")
        omega_ref_vals = [-omega_max, omega_max]
        for omega_ref in omega_ref_vals:
            s = copy.copy(self.nominal_params)
            s["omega_ref"] = omega_ref

            self.scenarios.append(s)

        self.P = torch.from_numpy(
            np.asarray(ast.literal_eval(config_data.get(section_name, "P")))
        )

        dynamics_section_name = config_data.get(section_name, "clbf_dynamics")
        self.clbf_dynamics = factory_from_config(
            dynamics_factory_base, config_data, dynamics_section_name
        )

    def V_with_jacobian(self, x: torch.Tensor):
        """Computes the CLF value and its Jacobian

        args:
            x: bs x n_state_dims the points at which to evaluate the CLF
        returns:
            V: bs tensor of CLF values
            JV: bs x 1 x n_state_dims Jacobian of each row of V wrt x
        """
        # First, get the Lyapunov function value and gradient at this state.
        # Use a quadratic Lyapunov function for now.
        P = self.P.type_as(x)
        # Reshape to use pytorch's bilinear function
        n_state_dims = 9  # for autorally CLBF
        P = P.reshape(1, n_state_dims, n_state_dims)
        V = 0.5 * F.bilinear(x, x, P).squeeze()
        V = V.reshape(x.shape[0])

        # Reshape again for the gradient calculation
        P = P.reshape(n_state_dims, n_state_dims)
        JV = F.linear(x, P)
        JV = JV.reshape(x.shape[0], 1, n_state_dims)

        return V, JV

    def V(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the value of the CLF"""
        V, _ = self.V_with_jacobian(x)
        return V

    def V_np(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        x = self.dynamics.surrogate_state_to_clbf_state(x).T

        return self.V(torch.from_numpy(x)).cpu().detach().numpy()

    def V_lie_derivatives(self, x: torch.Tensor):
        """Compute the Lie derivatives of the CLF V along the control-affine dynamics

        args:
            x: bs x self.dynamics.get_state_dim() tensor of state
        returns:
            Lf_V: bs x len(self.scenarios) x 1 tensor of Lie derivatives of V
                  along f
            Lg_V: bs x len(self.scenarios) x self.dynamics.get_action_dim() tensor
                  of Lie derivatives of V along g
        """
        n_scenarios = len(self.scenarios)

        # Get the Jacobian of V for each entry in the batch
        _, gradV = self.V_with_jacobian(x)

        # We need to compute Lie derivatives for each scenario
        batch_size = x.shape[0]
        Lf_V = torch.zeros(batch_size, n_scenarios, 1)
        Lg_V = torch.zeros(batch_size, n_scenarios, self.dynamics.get_action_dim()[0])
        Lf_V = Lf_V.type_as(x)
        Lg_V = Lg_V.type_as(x)

        for i in range(n_scenarios):
            # Get the dynamics f and g for this scenario
            s = self.scenarios[i]
            f, g = self.clbf_dynamics.torch_control_affine_dynamics(x, s)

            # Multiply these with the Jacobian to get the Lie derivatives
            Lf_V[:, i, :] = torch.bmm(gradV, f).squeeze(1)
            Lg_V[:, i, :] = torch.bmm(gradV, g).squeeze(1)

        # return the Lie derivatives
        return Lf_V, Lg_V

    def _solve_CLF_QP_gurobi(
        self,
        x: torch.Tensor,
        u_ref: torch.Tensor,
        V: torch.Tensor,
        Lf_V: torch.Tensor,
        Lg_V: torch.Tensor,
        relaxation_penalty: float,
    ):
        """Determine the control input for a given state using a QP. Solves the QP using
        Gurobi, which does not allow for backpropagation.

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            u_ref: bs x self.dynamics_model.n_controls tensor of reference controls
            V: bs x 1 tensor of CLF values,
            Lf_V: bs x 1 tensor of CLF Lie derivatives,
            Lg_V: bs x self.dynamics_model.n_controls tensor of CLF Lie derivatives,
            relaxation_penalty: the penalty to use for CLF relaxation.
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        # To find the control input, we want to solve a QP constrained by
        # L_f V + L_g V u + lambda V <= 0
        # To ensure that this QP is always feasible, we relax the constraint
        # L_f V + L_g V u + lambda V - r <= 0
        #                              r >= 0
        # and add the cost term relaxation_penalty * r.
        # We want the objective to be to minimize
        #           ||u - u_ref||^2 + relaxation_penalty * r^2
        # This reduces to (ignoring constant terms)
        #           u^T I u - 2 u_ref^T u + relaxation_penalty * r^2

        n_controls = self.dynamics.get_action_dim()[0]
        n_scenarios = len(self.scenarios)
        allow_relaxation = not (relaxation_penalty == float("inf"))

        # Solve a QP for each row in x
        bs = x.shape[0]
        u_result = torch.zeros(bs, n_controls)
        r_result = torch.zeros(bs, n_scenarios)
        for batch_idx in range(bs):
            # Skip any bad points
            if (
                torch.isnan(x[batch_idx]).any()
                or torch.isinf(x[batch_idx]).any()
                or torch.isnan(Lg_V[batch_idx]).any()
                or torch.isinf(Lg_V[batch_idx]).any()
                or torch.isnan(Lf_V[batch_idx]).any()
                or torch.isinf(Lf_V[batch_idx]).any()
            ):
                continue

            # Instantiate the model
            model = gp.Model("clf_qp")
            # Create variables for control input and (optionally) the relaxations
            upper_lim = self.dynamics.get_max_action()
            lower_lim = - upper_lim
            u = model.addMVar(n_controls, lb=lower_lim, ub=upper_lim)
            if allow_relaxation:
                r = model.addMVar(n_scenarios, lb=0, ub=GRB.INFINITY)

            # Define the cost
            Q = np.eye(n_controls)
            u_ref_np = u_ref[batch_idx, :].detach().cpu().numpy()
            objective = u @ Q @ u - 2 * u_ref_np @ Q @ u + u_ref_np @ Q @ u_ref_np
            if allow_relaxation:
                relax_penalties = relaxation_penalty * np.ones(n_scenarios)
                objective += relax_penalties @ r

            # Now build the CLF constraints
            for i in range(n_scenarios):
                Lg_V_np = Lg_V[batch_idx, i, :].detach().cpu().numpy()
                Lf_V_np = Lf_V[batch_idx, i, :].detach().cpu().numpy()
                V_np = V[batch_idx].detach().cpu().numpy()
                clf_constraint = Lf_V_np + Lg_V_np @ u + self.clbf_lambda * V_np
                if allow_relaxation:
                    clf_constraint -= r[i]
                model.addConstr(clf_constraint <= 0.0, name=f"Scenario {i} Decrease")

            # Optimize!
            model.setParam("LogToConsole", 0)
            model.setParam("DualReductions", 0)
            model.setObjective(objective, GRB.MINIMIZE)
            model.optimize()

            if model.status != GRB.OPTIMAL:
                print(f"WARNING: optimization status: {model.status}")
                # Make the relaxations nan if the problem was infeasible, as a signal
                # that something has gone wrong
                if allow_relaxation:
                    for i in range(n_scenarios):
                        r_result[batch_idx, i] = torch.tensor(float("nan"))
                continue

            # Extract the results
            for i in range(n_controls):
                u_result[batch_idx, i] = torch.tensor(u[i].x)
            if allow_relaxation:
                for i in range(n_scenarios):
                    r_result[batch_idx, i] = torch.tensor(r[i].x)

        return u_result.type_as(x), r_result.type_as(x)

    def solve_CLF_QP(
        self,
        x: torch.Tensor,
        relaxation_penalty: float,
        u_ref: torch.Tensor,
    ):
        """Determine the control input for a given state using a QP

        args:
            x: bs x self.dynamics_model.n_dims tensor of state
            relaxation_penalty: the penalty to use for CLF relaxation
            u_ref: allows the user to supply a custom reference input, which will
                   bypass the self.u_reference function. Must have
                   dimensions bs x self.dynamics_model.n_controls.
            requires_grad: if True, use a differentiable layer
        returns:
            u: bs x self.dynamics_model.n_controls tensor of control inputs
            relaxation: bs x 1 tensor of how much the CLF had to be relaxed in each
                        case
        """
        # Get the value of the CLF and its Lie derivatives
        V = self.V(x)
        Lf_V, Lg_V = self.V_lie_derivatives(x)

        return self._solve_CLF_QP_gurobi(x, u_ref, V, Lf_V, Lg_V, relaxation_penalty)

    def filter_control(
        self,
        state: np.ndarray,
        control: np.ndarray,
    ):
        """Filter the provided control action using the CLBF

        args:
            state: n_clbf_states array of states
            state: n_clbf_states array of states
        """
        # Convert the state to torch
        state = state.reshape(-1, 1)
        clbf_state = self.clbf_dynamics.surrogate_state_to_clbf_state(state)
        clbf_state = torch.from_numpy(clbf_state).T

        # Convert the control as well
        control = torch.from_numpy(control).reshape(1, -1)

        # Filter the control
        filtered_control, _ = self.solve_CLF_QP(
            clbf_state, self.clbf_relaxation_penalty, control
        )
        # filtered_control = control

        # print("----------")
        # print(f"    ar state: {state.T}")
        # print(f"  clbf state: {clbf_state.numpy()}")
        # print(f"MPPI control: {control.numpy()}")
        # print(f"CLBF control: {filtered_control.squeeze().numpy()}")

        return filtered_control.squeeze().numpy()

    def convert_to_surrogate_state(self, cartesian_state, control):
        """cartesian state:
        [vx, vy, wz, wF, wR, psi, X, Y]
        surrogate state:
        [delta, s_cumulative, vx, vy, wz, wF, wR, psi, X, Y]
        """
        delta = control[0, :]
        _, _, s_cumulative = self.dynamics.track.localize(cartesian_state[-2:, :], cartesian_state[5, :].reshape((1, -1)))
        surrogate_state = np.vstack((delta, s_cumulative, cartesian_state))
        return surrogate_state

    def plan(self, state_cur, warm_start=False):
        # print('state: ', state_cur)
        u = MPPI.plan(self, state_cur, warm_start)
        u[u > 1.0] = 1.0
        u[u < -1.0] = -1.0
        # print('control: ', u)
        state_clbf = self.convert_to_surrogate_state(state_cur.reshape((-1, 1)), u.reshape((-1, 1)))
        dt_wR = self.dynamics.throttle_model_no_nn(u[1:2].reshape((1, 1)), state_cur[4:5].reshape((-1, 1)))
        ref_clbf_control = np.hstack((0.0, 0.0))
        u_clbf = self.filter_control(state_clbf, ref_clbf_control)
        filtered_control = u.copy()
        filtered_control[0] += self.clbf_dynamics.get_delta_t() * self.clbf_dynamics.steering_factor * u_clbf[0]

        # r_wheel_acceleration = u_clbf[1:2]
        # r_wheel_speed = state_cur[4:5]
        # with torch.no_grad():
        #     throttle = self.clbf_dynamics.inverse_throttle_model(
        #         torch.from_numpy(r_wheel_speed).reshape(-1, 1).float(),
        #         torch.from_numpy(r_wheel_acceleration).reshape(-1, 1).float(),
        #     )
        # filtered_control[1] = throttle.squeeze()

        # print('filtered_control: ', filtered_control)
        return filtered_control
