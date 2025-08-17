import ipdb
import loguru

from robot_planning.controllers.MPPI.stochastic_trajectories_sampler import MPPIStochasticTrajectoriesSampler
import einops as ei
import numpy as onp
from robot_planning.controllers.controller import MpcController
# from robot_planning.environment.renderers import AutorallyMatplotlibRenderer
from robot_planning.environment.mpl_renderer import AutorallyMatplotlibRenderer
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import cost_evaluator_factory_base
from robot_planning.factory.factories import (
    stochastic_trajectories_sampler_factory_base,
)
import time
from robot_planning.environment.barriers.barrier_net import BarrierNN
import jax
import jax.numpy as np
from jax.scipy.optimize import minimize
from functools import partial
import ast
import copy
import scipy.optimize as sciopt

from robot_planning.helper.timer import Timer
import functools as ft


class MPPI(MpcController):
    def __init__(
        self,
        control_horizon=None,
        dynamics=None,
        cost_evaluator=None,
        control_dim=None,
        inverse_temperature=None,
        initial_control_sequence=None,
        stochastic_trajectories_sampler: MPPIStochasticTrajectoriesSampler = None,
        renderer=None,
    ):
        MpcController.__init__(
            self, control_horizon, dynamics, cost_evaluator, control_dim, renderer
        )
        self.inverse_temperature = inverse_temperature
        self.initial_control_sequence = initial_control_sequence
        self.stochastic_trajectories_sampler = stochastic_trajectories_sampler

        self.barrier_nn = BarrierNN()
        self.T_x_opt = None
        self.T_u_opt = None
        self.bT_rollout = None
        self.bT_us = None
        self.store_opt_traj = False

    def initialize_from_config(self, config_data, section_name):
        MpcController.initialize_from_config(self, config_data, section_name)
        self.inverse_temperature = config_data.getfloat(
            section_name, "inverse_temperature"
        )
        if config_data.has_option(section_name, "initial_control_sequence"):
            self.initial_control_sequence = np.asarray(
                ast.literal_eval(
                    config_data.get(section_name, "initial_control_sequence")
                ),
                dtype=np.float64,
            ).reshape((self.get_control_dim(), self.get_control_horizon() - 1))
            if not (
                self.initial_control_sequence.shape[0] is self.get_control_dim()
                and self.initial_control_sequence.shape[1]
                is self.get_control_horizon() - 1
            ):
                raise ValueError(
                    "The initial control sequence does not match control dimensions and control horizon"
                )
        else:
            self.initial_control_sequence = np.zeros(
                (self.get_control_dim(), self.get_control_horizon() - 1)
            )

        stochastic_trajectories_sampler_section_name = config_data.get(
            section_name, "stochastic_trajectories_sampler"
        )
        self.stochastic_trajectories_sampler = factory_from_config(
            stochastic_trajectories_sampler_factory_base,
            config_data,
            stochastic_trajectories_sampler_section_name,
        )
        if config_data.has_option(section_name,"repair_horizon"):
            self.repair_horizon = config_data.getint(
                section_name, "repair_horizon"
            )
        else:
            self.repair_horizon = None
        if config_data.has_option(section_name, "repair_steps"):
            self.repair_steps = config_data.getint(
                section_name, "repair_steps"
            )
        else:
            self.repair_steps = None

        if config_data.has_option(section_name, "min_controls"):
            self.min_controls = np.asarray(
                ast.literal_eval(
                    config_data.get(section_name, "min_controls")
                ), dtype=np.float32)
        else:
            self.min_controls = None
        if config_data.has_option(section_name, "max_controls"):
            self.max_controls = np.asarray(
                ast.literal_eval(
                    config_data.get(section_name, "max_controls")
                ), dtype=np.float32)
        else:
            self.max_controls = None
        if (self.min_controls is not None) and (self.max_controls is not None):
            self.control_bounds = np.asarray([self.min_controls, self.max_controls])
        else:
            self.control_bounds = None


    def plan(self, state_cur, warm_start=False, opponent_agents=None, logger=None):
        timer = Timer.get_active()
        # min_controls = [-1.0, -0.1]
        # max_controls = [1.0, 0.4]
        # control_bounds = [min_controls, max_controls]
        # control_bounds = np.asarray(control_bounds)

        timer_ = timer.child("copy").start()
        v = copy.deepcopy(self.initial_control_sequence)
        warm_start_itr = self.warm_start_itr if warm_start else 1
        timer_.stop()
        timer_ = timer.child("stoch traj sampler sample").start()
        for _ in range(warm_start_itr):
            # start = time.perf_counter()
            trajectories, us, costs = self.stochastic_trajectories_sampler.sample(
                state_cur,
                v,
                self.get_control_horizon(),
                self.get_control_dim(),
                self.dynamics,
                self.cost_evaluator,
                control_bounds=self.control_bounds,
                opponent_agents=opponent_agents
            )
            # print("Trajectory sampling frequency: ", 1/(time.perf_counter() - start))
            beta = np.min(costs)
            eta = np.sum(np.exp(-1 / self.inverse_temperature * (costs - beta)))
            omega = 1 / eta * np.exp(-1 / self.inverse_temperature * (costs - beta))
            # logger.add_omega(omega)

            v = np.sum(
                omega.reshape((us.shape[0], 1, 1)) * us, axis=0
            )  # us shape = (number_of_trajectories, control_dim, control_horizon)
            self.set_initial_control_sequence(v)
        timer_.stop()
        start = time.perf_counter()
        if self.repair_horizon:  # if we use CBF to carry out local repair
            timer_ = timer.child("local repair").start()
            v_safe = self.local_repair(v, state_cur)

            # # TODO Eric
            # vio_info_v = self.compute_ncbf_violation(v, state_cur)
            # vio_info_vsafe = self.compute_ncbf_violation(v_safe, state_cur)
            # logger.misc["h_vio_v"].append(vio_info_v)
            # logger.misc["h_vio_vsafe"].append(vio_info_vsafe)

            timer_.stop()
            # print("Local repair frequency: ", 1 / (time.perf_counter() - start))
        else:  # if original MPPI
            v_safe = v

        u = v_safe[:, 0]
        # Control Bounds
        if self.control_bounds is not None:
            u = np.clip(u, self.min_controls, self.max_controls)

        if self.renderer is not None:
            timer_ = timer.child("render_traj").start()
            if isinstance(self.renderer, AutorallyMatplotlibRenderer) and not self.renderer.trajectories_rendering:
                pass
            else:
                T_x_v = self.rollout_out(state_cur, v)

                self.renderer.render_trajectories(trajectories)
                self.renderer.render_trajectories([T_x_v])

                if self.repair_horizon:
                    # Render vsafe only if repair is performed and v != vsafe.
                    T_x_vsafe = self.rollout_out(state_cur, v_safe)
                    self.renderer.render_trajectories([T_x_vsafe], trajtype="vsafe")
            timer_.stop()
        elif self.store_opt_traj:
            bxT_rollout = onp.array(trajectories)
            bT_rollout = ei.rearrange(bxT_rollout, "b nx T -> b T nx")

            buT_us = onp.array(us)
            bT_us = ei.rearrange(buT_us, "b nu T -> b T nu")

            self.T_u_opt = v
            self.T_x_opt = self.rollout_out(state_cur, v)
            self.bT_rollout = bT_rollout
            self.bT_us = bT_us

        timer_ = timer.child("del hstack").start()
        v = np.delete(v, 0, 1)
        v = np.hstack((v, v[:, -1].reshape(v.shape[0], 1)))
        timer_.stop()
        self.set_initial_control_sequence(v)
        return u

    def reset(self):
        self.initial_control_sequence = np.zeros(
            (self.get_control_dim(), self.get_control_horizon() - 1)
        )

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def repair_cost(self, control, state, control_shape, rollout_horizon):
        print("re-tracing repair cost")
        cost = 0
        state = state.reshape(-1, 1)
        control = control.reshape(control_shape)

        # Loop over the rollout horizon using LAX scan
        controls = control.T  # transpose so leading axis is timestep
        initial_carry = (0.0, state)  # cost and current state
        def scan_fn(carry, control):
            running_cost, state = carry
            state_next = self.dynamics.propagate(state, control.reshape(-1, 1)).reshape(-1, 1)
            running_cost += self.cost_evaluator.evaluate_cbf_cost(
                state, dynamics=self.dynamics, state_next=state_next
            )
            return (running_cost, state_next), None

        final_carry, _ = jax.lax.scan(scan_fn, initial_carry, controls)
        cost, _ = final_carry

        print("done tracing repair cost")

        return cost

    @partial(jax.jit, static_argnums=(0,))
    def local_repair(self, v, state_cur):
        if self.repair_steps <= 0:
            print("Not performing repair.")
            return v

        print(f"re-tracing repair fn")
        v = np.array(v[:, :self.repair_horizon])

        result = minimize(
            self.repair_cost,
            v.reshape(-1),
            args=(state_cur, v.shape, self.repair_horizon),
            method="BFGS",
            options={"maxiter": self.repair_steps}
        )

        # # TODO Eric
        # repair_cost_old = self.repair_cost(v.reshape(-1), state_cur, v.shape, self.repair_horizon)
        # repair_cost_new = self.repair_cost(result.x, state_cur, v.shape, self.repair_horizon)

        v_new = result.x

        # # TODO Eric
        # got_worse = repair_cost_new > repair_cost_old
        # v_new = np.where(got_worse, v.reshape(-1), v_new)

        print("done tracing repair fn, but JIT compilation may take awhile...")

        return v_new.reshape(v.shape)
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_ncbf_violation(self, v, state_cur):
        # state_cur: (nx, 1)
        # v: (nu, T)
        # control: (T, nu)
        control = v.T
        # control0: (nu, )
        control0 = control[0]
        # (nx, 1)
        state_next = self.dynamics.propagate(state_cur, control0.reshape(-1, 1)).reshape(-1, 1)

        state_cur_sane = state_cur
        state_next_sane = state_next.squeeze(1)

        cost_fn = ft.partial(self.cost_evaluator.evaluate_cost, dynamics=self.dynamics)
        cost_run, unsafe, info = cost_fn(state_cur_sane, state_next_sane, control0, control0)

        h_vio = info["h_vio"]
        h_h_curr = info["h_h_curr"]
        h_h_next = info["h_h_next"]
        h_Vh_curr = info["h_Vh_curr"]
        h_Vh_next = info["h_Vh_next"]

        cbf_cost = self.cost_evaluator.evaluate_cbf_cost(state_cur[:, None], dynamics=self.dynamics, state_next=state_next)

        out = {"h_vio": h_vio, "h_h_curr": h_h_curr, "h_h_next": h_h_next, "cbf_cost": cbf_cost, "h_Vh_curr": h_Vh_curr, "h_Vh_next": h_Vh_next}
        return out

    def set_initial_control_sequence(self, initial_control_sequence):
        self.initial_control_sequence = initial_control_sequence

    def rollout_out(self, state_cur, v):
        trajectory = np.zeros((self.dynamics.get_state_dim()[0], v.shape[1] + 1))
        trajectory = trajectory.at[:, 0].set(state_cur)
        for i in range(v.shape[1]):
            state_next = self.dynamics.propagate(state_cur.reshape(-1, 1), v[:, i].reshape(-1, 1)).reshape(state_cur.shape)
            trajectory = trajectory.at[:, i + 1].set(state_next)
            state_cur = state_next
        return trajectory


class CEMMPC(MPPI):
    def __init__(
        self,
        control_horizon=None,
        dynamics=None,
        cost_evaluator=None,
        control_dim=None,
        inverse_temperature=None,
        initial_control_sequence=None,
        stochastic_trajectories_sampler: MPPIStochasticTrajectoriesSampler = None,
        renderer=None,
    ):
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
    def initialize_from_config(self, config_data, section_name):
        MPPI.initialize_from_config(self, config_data, section_name)
        self.elite_ratio = config_data.getfloat(section_name, "elite_ratio")

    def top_k_indices_unordered(self, arr, top_percent=0.1):
        arr_flat = arr.flatten()  # Flatten the array
        k = max(1, int(len(arr_flat) * top_percent))  # Calculate bottom k elements (at least 1)

        # Get indices of the top k smallest values (unordered)
        top_k_idx = np.argpartition(arr_flat, -k)[-k:]

        return np.unravel_index(top_k_idx, arr.shape)[0]  # Convert flat indices back to original shape

    def plan(self, state_cur, warm_start=False, opponent_agents=None, logger=None):
        timer = Timer.get_active()
        # min_controls = [-1.0, -0.1]
        # max_controls = [1.0, 0.4]
        # control_bounds = [min_controls, max_controls]
        # control_bounds = np.asarray(control_bounds)

        timer_ = timer.child("copy").start()
        v = copy.deepcopy(self.initial_control_sequence)
        warm_start_itr = self.warm_start_itr if warm_start else 1
        timer_.stop()
        timer_ = timer.child("stoch traj sampler sample").start()
        for _ in range(warm_start_itr):
            # start = time.perf_counter()
            trajectories, us, costs = self.stochastic_trajectories_sampler.sample(
                state_cur,
                v,
                self.get_control_horizon(),
                self.get_control_dim(),
                self.dynamics,
                self.cost_evaluator,
                control_bounds=self.control_bounds,
                opponent_agents=opponent_agents
            )
            # print("Trajectory sampling frequency: ", 1/(time.perf_counter() - start))
            beta = np.min(costs)
            eta = np.sum(np.exp(-1 / self.inverse_temperature * (costs - beta)))
            omega = 1 / eta * np.exp(-1 / self.inverse_temperature * (costs - beta))  # Eric: Weights

            top_indices = self.top_k_indices_unordered(omega, top_percent=self.elite_ratio)

            mask = onp.zeros(omega.shape[0], dtype=float)  # Create a zeros mask
            mask[top_indices] = 1.0/len(top_indices)  # normalization
            # mask = mask.at[top_indices].set(1)  # Set top indices to 1
            # ipdb.set_trace()
            v = np.sum(
                mask.reshape((us.shape[0], 1, 1)) * us, axis=0
            )  # us shape = (number_of_trajectories, control_dim, control_horizon)
            self.set_initial_control_sequence(v)
        timer_.stop()
        start = time.perf_counter()
        if self.repair_horizon:  # if we use CBF to carry out local repair
            timer_ = timer.child("local repair").start()
            v_safe = self.local_repair(v, state_cur)
            timer_.stop()
            # print("Local repair frequency: ", 1 / (time.perf_counter() - start))
        else:  # if original MPPI
            v_safe = v

        u = v_safe[:, 0]
        # Control Bounds
        if self.control_bounds is not None:
            u = np.clip(u, self.min_controls, self.max_controls)

        if self.renderer is not None:
            timer_ = timer.child("render_traj").start()
            if isinstance(self.renderer, AutorallyMatplotlibRenderer) and not self.renderer.trajectories_rendering:
                pass
            else:
                T_x_opt = self.rollout_out(state_cur, v)
                self.renderer.render_trajectories(trajectories, **{"color": "b"})
                self.renderer.render_trajectories([T_x_opt], **{"color": "r"})
            timer_.stop()
        elif self.store_opt_traj:
            bxT_rollout = onp.array(trajectories)
            bT_rollout = ei.rearrange(bxT_rollout, "b nx T -> b T nx")

            buT_us = onp.array(us)
            bT_us = ei.rearrange(buT_us, "b nu T -> b T nu")

            self.T_u_opt = v
            self.T_x_opt = self.rollout_out(state_cur, v)
            self.bT_rollout = bT_rollout
            self.bT_us = bT_us

        timer_ = timer.child("del hstack").start()
        v = np.delete(v, 0, 1)
        v = np.hstack((v, v[:, -1].reshape(v.shape[0], 1)))
        timer_.stop()
        self.set_initial_control_sequence(v)
        return u