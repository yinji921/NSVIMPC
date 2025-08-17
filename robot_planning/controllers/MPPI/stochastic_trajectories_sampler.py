import ipdb
import jax.numpy as np
import jax.debug as jd
import jax
import jax.lax as lax
import ast
import copy
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import noise_sampler_factory_base
import multiprocessing as mp
from robot_planning.factory.factories import covariance_steering_helper_factory_base
from robot_planning.factory.factories import dynamics_factory_base
import threading
import time
from robot_planning.helper.common import PrintObject

from functools import partial
import functools as ft
import jax.random as jr
import jax.numpy as jnp
import einops as ei
from robot_planning.environment.dynamics.autorally_dynamics.autorally_dynamics import AutoRallyDynamics
from robot_planning.environment.cost_evaluators import AutorallyMPPICBFCostEvaluator
from typing import NamedTuple

class SamplerResult(NamedTuple):
    bxT_state: jnp.ndarray
    buT_control: jnp.ndarray
    b11_cost: jnp.ndarray


class DetailedSamplerResult(NamedTuple):
    result: SamplerResult
    info: dict

class StochasticTrajectoriesSampler(PrintObject):
    def __init__(
        self,
        number_of_trajectories=None,
        uncontrolled_trajectories_portion=None,
        noise_sampler=None,
    ):
        self.number_of_trajectories = number_of_trajectories
        self.uncontrolled_trajectories_portion = uncontrolled_trajectories_portion
        self.noise_sampler = noise_sampler

    def initialize_from_config(self, config_data, section_name):
        self.number_of_trajectories = int(
            config_data.getfloat(section_name, "number_of_trajectories")
        )
        self.uncontrolled_trajectories_portion = config_data.getfloat(
            section_name, "uncontrolled_trajectories_portion"
        )
        noise_sampler_section_name = config_data.get(section_name, "noise_sampler")
        self.noise_sampler = factory_from_config(
            noise_sampler_factory_base, config_data, noise_sampler_section_name
        )

    def sample(
        self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator
    ):
        raise NotImplementedError

    def get_number_of_trajectories(self):
        return copy.copy(self.number_of_trajectories)

    def set_number_of_trajectories(self, number_of_trajectories):
        self.number_of_trajectories = number_of_trajectories


class MPPIStochasticTrajectoriesSamplerSlowLoop(StochasticTrajectoriesSampler):
    def __init__(
        self,
        number_of_trajectories=None,
        uncontrolled_trajectories_portion=None,
        noise_sampler=None,
    ):
        StochasticTrajectoriesSampler.__init__(
            self,
            number_of_trajectories,
            uncontrolled_trajectories_portion,
            noise_sampler,
        )

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(
            self, config_data, section_name
        )

    def sample(
        self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator
    ):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur.reshape((-1, 1))
        us = np.zeros((self.number_of_trajectories, control_dim, control_horizon - 1))
        costs = np.zeros((self.number_of_trajectories, 1, 1))
        trajectories = []
        for i in range(self.number_of_trajectories):
            state_cur = state_start
            trajectory = np.zeros((dynamics.get_state_dim()[0], control_horizon))
            trajectory[:, :1] = state_cur
            noises = self.noise_sampler.sample(control_dim, control_horizon - 1)
            if (
                i
                > (1 - self.uncontrolled_trajectories_portion)
                * self.number_of_trajectories
            ):
                u = v + noises
            else:
                u = noises
            cost = 0
            for j in range(control_horizon - 1):
                cost += cost_evaluator.evaluate(
                    state_cur, u[:, j : j + 1], dynamics=dynamics
                )
                state_cur = dynamics.propagate(state_cur.reshape(-1, 1), u[:, j]).reshape((-1, 1))
                trajectory[:, j + 1 : j + 2] = state_cur
            cost += cost_evaluator.evaluate_terminal_cost(state_cur, dynamics=dynamics)
            trajectories.append(trajectory)
            us[i, :, :] = u
            costs[i, 0, 0] = cost
        return trajectories, us, costs


class MPPIStochasticTrajectoriesSampler(StochasticTrajectoriesSampler):
    def __init__(
        self,
        number_of_trajectories=None,
        uncontrolled_trajectories_portion=None,
        noise_sampler=None,
    ):
        StochasticTrajectoriesSampler.__init__(
            self,
            number_of_trajectories,
            uncontrolled_trajectories_portion,
            noise_sampler,
        )

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(
            self, config_data, section_name
        )

    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
    def sample(
        self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator, control_bounds=None, opponent_agents=None):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur.copy()
        costs = np.zeros((self.number_of_trajectories, 1, 1))
        state_cur = np.tile(
            state_start.reshape((-1, 1)), (1, self.number_of_trajectories)
        )
        trajectories = np.zeros(
            (self.number_of_trajectories, state_cur.shape[0], control_horizon)
        )
        trajectories = trajectories.at[:, :, 0].set(np.swapaxes(state_cur, 0, 1))
        noises = self.noise_sampler.sample(
            control_dim, (control_horizon - 1) * self.number_of_trajectories
        )
        noises = noises.reshape(
            (control_dim, (control_horizon - 1), self.number_of_trajectories)
        )
        num_controlled_trajectories = int(
            (1 - self.uncontrolled_trajectories_portion) * self.number_of_trajectories
        )
        us = np.zeros((v.shape[0], v.shape[1], self.number_of_trajectories))
        us = us.at[:, :, :num_controlled_trajectories].set(np.expand_dims(v, axis=2))
        us += noises
        # Control Bounds # TODO make the bounds in the config file
        if control_bounds is not None:
            us = us.at[0, :, :].set(np.where(us[0, :, :] < control_bounds[1, 0], us[0, :, :], control_bounds[1, 0]))  # upper bound on steering
            us = us.at[0, :, :].set(np.where(us[0, :, :] > control_bounds[0, 0], us[0, :, :], control_bounds[0, 0]))  # lower bound on steering
            us = us.at[1, :, :].set(np.where(us[1, :, :] < control_bounds[1, 1], us[1, :, :], control_bounds[1, 1]))  # upper bound on throttle
            us = us.at[1, :, :].set(np.where(us[1, :, :] > control_bounds[0, 1], us[1, :, :], control_bounds[0, 1]))  # lower bound on throttle

        # start = time.time()
        for j in range(control_horizon - 1):
            costs += cost_evaluator.evaluate(
                state_cur, us[:, j, :], noises[:, j, :], dynamics=dynamics, opponent_agents=opponent_agents
            )
            state_cur = dynamics.propagate(state_cur, us[:, j, :])
            # ipdb.set_trace()
            trajectories = trajectories.at[:, :, j + 1].set(np.swapaxes(state_cur, 0, 1))
        costs += cost_evaluator.evaluate_terminal_cost(state_cur, dynamics=dynamics)
        # jax.debug.print("dt = {dt}", dt = time.time() - start)
        us = np.moveaxis(us, 2, 0)
        return trajectories, us, costs


class MPPICBFStochasticTrajectoriesSampler(MPPIStochasticTrajectoriesSampler):
    def __init__(
        self,
        number_of_trajectories=None,
        uncontrolled_trajectories_portion=None,
        noise_sampler=None,
    ):
        MPPIStochasticTrajectoriesSampler.__init__(
            self,
            number_of_trajectories,
            uncontrolled_trajectories_portion,
            noise_sampler,
        )

    def initialize_from_config(self, config_data, section_name):
        MPPIStochasticTrajectoriesSampler.initialize_from_config(
            self, config_data, section_name
        )
        if config_data.has_option(section_name, "efficient_sampling"):
            self.efficient_sampling = config_data.getboolean(section_name, "efficient_sampling")
        else:
            self.efficient_sampling = False

    @property
    def n_traj(self):
        return self.number_of_trajectories

    def sample(
        self,
        state_cur,
        v,
        control_horizon: int,
        control_dim: int,
        dynamics: AutoRallyDynamics,
        cost_evaluator: AutorallyMPPICBFCostEvaluator,
        control_bounds=None,
        opponent_agents=None,
    ):
        if self.efficient_sampling:
            detailed_result = self._sample_efficient(
                state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator, control_bounds
            )
            return detailed_result.result
        else:
            detailed_result = self._sample(
                state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator, control_bounds
            )
            return detailed_result.result


    @ft.partial(jax.jit, static_argnames=("self", "control_horizon", "control_dim", "dynamics", "cost_evaluator"))
    def _sample_efficient(
            self,
            state_cur,
            v,
            control_horizon: int,
            control_dim: int,
            dynamics: AutoRallyDynamics,
            cost_evaluator: AutorallyMPPICBFCostEvaluator,
            control_bounds=None,
    ) -> DetailedSamplerResult:
        """
        :param state_cur: Current state. (nx, ) = (8, )
        :param v: Current nominal control sequence. (nu, control_horizon - 1) = (2, 19)
        :param control_horizon:
        :param control_dim:
        :param dynamics:
        :param cost_evaluator:
        :param control_bounds: (2, nu) = (2, 2)
        :return:
        """
        uT_v = v

        assert self.uncontrolled_trajectories_portion == 0.0
        # (nu, (control_horizon - 1) * n_trajs)
        ub_noise = self.noise_sampler.sample(control_dim, (control_horizon - 1) * self.n_traj)
        Tbu_noise = ei.rearrange(ub_noise, "nu (T b) -> T b nu", T=control_horizon - 1, b=self.n_traj)

        Tu_v = ei.rearrange(uT_v, "nu T -> T nu")
        Tbu_control = Tbu_noise + Tu_v[:, None, :]
        if control_bounds is not None:
            lb, ub = control_bounds[0], control_bounds[1]
            Tbu_control = jnp.clip(Tbu_control, lb, ub)

        # Rollout.
        bx_state0 = ei.repeat(state_cur, "nx -> b nx", b=self.n_traj)
        vmap_prop = jax.vmap(dynamics.propagate)

        #   cost fns
        cost_fn = ft.partial(cost_evaluator.evaluate_cost, dynamics=dynamics)
        term_cost_fn = ft.partial(cost_evaluator.evaluate_terminal_cost, dynamics=dynamics)

        def body(trajstate, bu_controlnoise):
            bx_state, b_cost, Tp1bx_state_, Tbu_control_ = trajstate
            ii, bu_control, bu_noise = bu_controlnoise

            bx_state_next = vmap_prop(bx_state, bu_control)

            b_cost_run, b_unsafe, info = jax.vmap(cost_fn)(bx_state, bx_state_next, bu_control, bu_noise)
            b_cost_next = b_cost + b_cost_run

            # bh_Vh_next = info["h_Vh_next"]
            bh_h_next = info["h_h_next"]
            # b_Vh_next = jnp.max(bh_Vh_next, axis=1)
            b_h_next = jnp.max(bh_h_next, axis=1)
            assert b_h_next.shape == (self.n_traj,)

            # If all unsafe, then make the top k lowest h_Vh_next safe.
            # b_Vh_next_argsort = jnp.argsort(b_Vh_next, axis=0)
            b_Vh_next_argsort = jnp.argsort(b_h_next, axis=0)
            k_frac = 0.1
            k = int(k_frac * self.n_traj)
            b_unsafe_topk = b_Vh_next_argsort > k

            all_unsafe = jnp.all(b_unsafe, axis=0)
            b_unsafe = jnp.where(all_unsafe, b_unsafe_topk, b_unsafe)

            # Any trajs that are unsafe should be resampled from trajs that are safe.
            b_prob = jnp.where(b_unsafe, 0.05, 1.0)
            b_prob = b_prob / b_prob.sum()

            # b_prob = jnp.where(all_unsafe, 1.0, b_prob)

            key = jr.PRNGKey(12345 + ii)
            b_idx_new = jr.choice(key, self.n_traj, shape=(self.n_traj,), p=b_prob)

            # If it is safe, then keep the original idx.
            b_idx_new = jnp.where(b_unsafe, b_idx_new, jnp.arange(self.n_traj))
            # b_idx_new = jnp.arange(self.n_traj)

            # First, set the control and histories to the original, non-resampled versions.
            Tbu_control_ = Tbu_control_.at[ii].set(bu_control)
            Tp1bx_state_ = Tp1bx_state_.at[ii + 1].set(bx_state_next)

            # Next, replace the state, cost and histories with the new ones.
            bx_state_next_resam = bx_state_next[b_idx_new]
            b_cost_next_resam = b_cost_next[b_idx_new]

            Tp1bx_state_resam = Tp1bx_state_[:, b_idx_new, :]
            Tbu_control_resam = Tbu_control_[:, b_idx_new, :]

            info_ = {
                "p_safe": 1.0 - jnp.mean(b_unsafe),
            }

            trajstate_new = (bx_state_next_resam, b_cost_next_resam, Tp1bx_state_resam, Tbu_control_resam)
            return trajstate_new, info_

        Tp1bx_state = ei.repeat(bx_state0, "b nx -> T b nx", T=control_horizon)
        trajstate0 = (bx_state0, jnp.zeros((self.n_traj, 1, 1)), Tp1bx_state, Tbu_control)
        inp = jnp.arange(control_horizon - 1), Tbu_control, Tbu_noise
        trajstate, scan_info = lax.scan(body, trajstate0, inp, length=control_horizon - 1, unroll=8)

        _, b11_cost, Tp1bx_state, Tbu_control_new = trajstate
        del Tbu_control

        # (control_horizon, batch, nx)
        Tbx_state_from = Tp1bx_state[:-1]
        Tbx_state_to = Tp1bx_state[1:]
        bx_state_last = Tp1bx_state[-1]

        # cost_fn returns (1, 1)
        Tb11_cost, _, cost_info = jax.vmap(jax.vmap(cost_fn))(Tbx_state_from, Tbx_state_to, Tbu_control_new, Tbu_noise)
        b11_cost = jnp.sum(Tb11_cost, axis=0)

        #       This assumes a batch dimension at the end.
        b_cost_term = jax.vmap(term_cost_fn)(bx_state_last[:, :, None]).squeeze(-1)
        b11_cost = b11_cost + b_cost_term
        assert b11_cost.shape == (self.n_traj, 1, 1)

        # Reshape for output.
        bxT_state = ei.rearrange(Tp1bx_state, "T b nx -> b nx T")
        buT_control = ei.rearrange(Tbu_control_new, "T b nu -> b nu T")

        T_psafe = scan_info["p_safe"]
        info = {"T_psafe": T_psafe}

        result = SamplerResult(bxT_state, buT_control, b11_cost)
        return DetailedSamplerResult(result, info)

    @ft.partial(jax.jit, static_argnames=("self", "control_horizon", "control_dim", "dynamics", "cost_evaluator"))
    def _sample(
            self,
            state_cur,
            v,
            control_horizon: int,
            control_dim: int,
            dynamics: AutoRallyDynamics,
            cost_evaluator: AutorallyMPPICBFCostEvaluator,
            control_bounds=None,
    ) -> DetailedSamplerResult:
        """
        :param state_cur: Current state. (nx, ) = (8, )
        :param v: Current nominal control sequence. (nu, control_horizon - 1) = (2, 19)
        :param control_horizon:
        :param control_dim:
        :param dynamics:
        :param cost_evaluator:
        :param control_bounds: (2, nu) = (2, 2)
        :return:
        """
        uT_v = v

        assert self.uncontrolled_trajectories_portion == 0.0
        # (nu, (control_horizon - 1) * n_trajs)
        ub_noise = self.noise_sampler.sample(control_dim, (control_horizon - 1) * self.n_traj)
        Tbu_noise = ei.rearrange(ub_noise, "nu (T b) -> T b nu", T=control_horizon - 1, b=self.n_traj)

        Tu_v = ei.rearrange(uT_v, "nu T -> T nu")
        Tbu_control = Tbu_noise + Tu_v[:, None, :]
        if control_bounds is not None:
            lb, ub = control_bounds[0], control_bounds[1]
            Tbu_control = jnp.clip(Tbu_control, lb, ub)

        # Rollout.
        bx_state0 = ei.repeat(state_cur, "nx -> b nx", b=self.n_traj)
        vmap_prop = jax.vmap(dynamics.propagate)

        #   cost fns
        cost_fn = ft.partial(cost_evaluator.evaluate_cost, dynamics=dynamics)
        term_cost_fn = ft.partial(cost_evaluator.evaluate_terminal_cost, dynamics=dynamics)

        def body(trajstate, bu_controlnoise):
            bx_state, b_cost, Tp1bx_state_, Tbu_control_ = trajstate
            ii, bu_control, bu_noise = bu_controlnoise
            bx_state_next = vmap_prop(bx_state, bu_control)

            b_cost_run, b_unsafe, info = jax.vmap(cost_fn)(bx_state, bx_state_next, bu_control, bu_noise)
            b_cost_next = b_cost + b_cost_run

            # Set the control and histories to the original, non-resampled versions.
            Tbu_control_ = Tbu_control_.at[ii].set(bu_control)
            Tp1bx_state_ = Tp1bx_state_.at[ii + 1].set(bx_state_next)

            info_ = {
                "p_safe": 1.0 - jnp.mean(b_unsafe),
            }

            trajstate_new = (bx_state_next, b_cost_next, Tp1bx_state_, Tbu_control_)
            return trajstate_new, info_

        Tp1bx_state = ei.repeat(bx_state0, "b nx -> T b nx", T=control_horizon)
        trajstate0 = (bx_state0, jnp.zeros((self.n_traj, 1, 1)), Tp1bx_state, Tbu_control)
        inp = jnp.arange(control_horizon - 1), Tbu_control, Tbu_noise
        trajstate, scan_info = lax.scan(body, trajstate0, inp, length=control_horizon - 1, unroll=8)

        _, b11_cost, Tp1bx_state, Tbu_control_new = trajstate
        del Tbu_control

        bx_state_last = Tp1bx_state[-1]
        #       This assumes a batch dimension at the end.
        b_cost_term = jax.vmap(term_cost_fn)(bx_state_last[:, :, None]).squeeze(-1)
        b11_cost = b11_cost + b_cost_term
        assert b11_cost.shape == (self.n_traj, 1, 1)

        # Reshape for output.
        bxT_state = ei.rearrange(Tp1bx_state, "T b nx -> b nx T")
        buT_control = ei.rearrange(Tbu_control_new, "T b nu -> b nu T")

        T_psafe = scan_info["p_safe"]
        info = {"T_psafe": T_psafe}

        result = SamplerResult(bxT_state, buT_control, b11_cost)
        return DetailedSamplerResult(result, info)

    # @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
    # def sample(
    #     self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator, control_bounds=None, opponent_agents=None
    # ):
    #     import ipdb
    #     #  state_cur is the current state, v is the nominal control sequence
    #     state_start = state_cur.copy()
    #     costs = np.zeros((self.number_of_trajectories, 1, 1))
    #     state_cur = np.tile(
    #         state_start.reshape((-1, 1)), (1, self.number_of_trajectories)
    #     )
    #     trajectories = np.zeros(
    #         (self.number_of_trajectories, state_cur.shape[0], control_horizon)
    #     )
    #     trajectories = trajectories.at[:, :, 0].set(np.swapaxes(state_cur, 0, 1))
    #     noises = self.noise_sampler.sample(
    #         control_dim, (control_horizon - 1) * self.number_of_trajectories
    #     )
    #     noises = noises.reshape(
    #         (control_dim, (control_horizon - 1), self.number_of_trajectories)
    #     )
    #     num_controlled_trajectories = int(
    #         (1 - self.uncontrolled_trajectories_portion) * self.number_of_trajectories
    #     )
    #     us = np.zeros((v.shape[0], v.shape[1], self.number_of_trajectories))
    #     us = us.at[:, :, :num_controlled_trajectories].set(np.expand_dims(v, axis=2))
    #     us += noises
    #     # Control Bounds # TODO make the bounds in the config file
    #     if control_bounds is not None:
    #         us = us.at[0, :, :].set(np.where(us[0, :, :] < control_bounds[1, 0], us[0, :, :], control_bounds[1, 0]))  # upper bound on steering
    #         us = us.at[0, :, :].set(np.where(us[0, :, :] > control_bounds[0, 0], us[0, :, :], control_bounds[0, 0]))  # lower bound on steering
    #         us = us.at[1, :, :].set(np.where(us[1, :, :] < control_bounds[1, 1], us[1, :, :], control_bounds[1, 1]))  # upper bound on throttle
    #         us = us.at[1, :, :].set(np.where(us[1, :, :] > control_bounds[0, 1], us[1, :, :], control_bounds[0, 1]))  # lower bound on throttle
    #     # #D ebug values
    #     # print(np.max(us, axis=2))
    #     # print(np.min(us, axis=2))
    #     propagation_time = 0.0
    #     for j in range(control_horizon - 1):
    #         start = time.perf_counter()
    #         state_next = dynamics.propagate(state_cur, us[:, j, :])
    #         propagation_time += time.perf_counter() - start
    #         costs += cost_evaluator.evaluate(
    #             state_cur, us[:, j, :], noises[:, j, :], dynamics=dynamics, state_next=state_next,
    #         )
    #         state_cur = state_next
    #         trajectories = trajectories.at[:, :, j + 1].set(np.swapaxes(state_cur, 0, 1))
    #     costs += cost_evaluator.evaluate_terminal_cost(state_cur, dynamics=dynamics)
    #     us = np.moveaxis(us, 2, 0)
    #     return trajectories, us, costs

class RAMPPIStochasticTrajectoriesSampler(MPPIStochasticTrajectoriesSampler):
    def __init__(
        self,
        number_of_trajectories=None,
        uncontrolled_trajectories_portion=None,
        noise_sampler=None,
    ):
        StochasticTrajectoriesSampler.__init__(
            self,
            number_of_trajectories,
            uncontrolled_trajectories_portion,
            noise_sampler,
        )

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(
            self, config_data, section_name
        )
        dynamics_section_name = config_data.get(section_name, "dynamics")
        self.disturbed_dynamics = factory_from_config(
            dynamics_factory_base, config_data, dynamics_section_name
        )
        self.number_of_CVAR_trajectories = config_data.getint(section_name, "number_of_CVAR_trajectories")
        self.confidence_level = config_data.getfloat(section_name, "confidence_level")
        self.CVaR_weight = config_data.getfloat(section_name, "CVaR_weight")

    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6))
    def sample(
        self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator, control_bounds=None, opponent_agents=None
    ):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur.copy()
        costs = np.zeros((self.number_of_trajectories, 1, 1))
        state_cur = np.tile(
            state_start.reshape((-1, 1)), (1, self.number_of_trajectories)
        )
        trajectories = np.zeros(
            (self.number_of_trajectories, state_cur.shape[0], control_horizon)
        )
        trajectories = trajectories.at[:, :, 0].set(np.swapaxes(state_cur, 0, 1))
        noises = self.noise_sampler.sample(
            control_dim, (control_horizon - 1) * self.number_of_trajectories
        )
        noises = noises.reshape(
            (control_dim, (control_horizon - 1), self.number_of_trajectories)
        )
        num_controlled_trajectories = int(
            (1 - self.uncontrolled_trajectories_portion) * self.number_of_trajectories
        )
        us = np.zeros((v.shape[0], v.shape[1], self.number_of_trajectories))
        us = us.at[:, :, :num_controlled_trajectories].set(np.expand_dims(v, axis=2))
        us += noises
        # Control Bounds # TODO make the bounds in the config file
        if control_bounds is not None:
            us = us.at[0, :, :].set(np.where(us[0, :, :] < control_bounds[1, 0], us[0, :, :], control_bounds[1, 0]))  # upper bound on steering
            us = us.at[0, :, :].set(np.where(us[0, :, :] > control_bounds[0, 0], us[0, :, :], control_bounds[0, 0]))  # lower bound on steering
            us = us.at[1, :, :].set(np.where(us[1, :, :] < control_bounds[1, 1], us[1, :, :], control_bounds[1, 1]))  # upper bound on throttle
            us = us.at[1, :, :].set(np.where(us[1, :, :] > control_bounds[0, 1], us[1, :, :], control_bounds[0, 1]))  # lower bound on throttle

        for j in range(control_horizon - 1):
            costs += cost_evaluator.evaluate(
                state_cur, us[:, j, :], noises[:, j, :], dynamics=dynamics
            )
            state_cur = dynamics.propagate(state_cur, us[:, j, :])
            trajectories = trajectories.at[:, :, j + 1].set(np.swapaxes(state_cur, 0, 1))
        costs += cost_evaluator.evaluate_terminal_cost(state_cur, dynamics=dynamics)

        # create the current state for risk evaluation
        risk_state_cur = np.tile(
            state_start.reshape((-1, 1)), (1, self.number_of_trajectories * self.number_of_CVAR_trajectories)
        )
        risk_us = np.tile(us, (1, 1,self.number_of_CVAR_trajectories)) #TODO: find the right dimension to pile us
        risk_trajectories = np.zeros(
            (self.number_of_trajectories * self.number_of_CVAR_trajectories, risk_state_cur.shape[0], control_horizon)
        )
        risk_trajectories = risk_trajectories.at[:, :, 0].set(np.swapaxes(risk_state_cur, 0, 1))
        risk_costs = np.zeros((self.number_of_trajectories * self.number_of_CVAR_trajectories, 1, 1))
        for j in range(control_horizon - 1):
            risk_costs += cost_evaluator.evaluate(
                risk_state_cur, risk_us[:, j, :], None, dynamics=self.disturbed_dynamics
            )
            risk_state_cur = self.disturbed_dynamics.propagate(risk_state_cur, risk_us[:, j, :])
            risk_trajectories = risk_trajectories.at[:, :, j + 1].set(np.swapaxes(risk_state_cur, 0, 1))
        risk_costs += cost_evaluator.evaluate_terminal_cost(risk_state_cur, dynamics=self.disturbed_dynamics)

        risk_costs = risk_costs.reshape((self.number_of_CVAR_trajectories, self.number_of_trajectories, 1, 1))
        risk_costs = np.sort(risk_costs, axis=0)
        # Compute the CVaR value for trajectory costs #
        risk_costs = np.average(risk_costs[-int((1.0 - self.confidence_level)*self.number_of_CVAR_trajectories):,:,:], axis=0)

        # Penalize the nominal trajectories with CVaR costs #
        # costs += self.CVaR_weight*risk_costs
        costs = self.CVaR_weight*risk_costs

        us = np.moveaxis(us, 2, 0)
        return trajectories, us, costs


class MPPIParallelStochasticTrajectoriesSamplerMultiprocessing(
    StochasticTrajectoriesSampler
):
    def __init__(
        self,
        number_of_trajectories=None,
        uncontrolled_trajectories_portion=None,
        noise_sampler=None,
        number_of_processes=8,
    ):
        StochasticTrajectoriesSampler.__init__(
            self,
            number_of_trajectories,
            uncontrolled_trajectories_portion,
            noise_sampler,
        )
        self.number_of_processes = 8

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(
            self, config_data, section_name
        )
        if config_data.has_option(section_name, "number_of_processes"):
            self.number_of_processes = config_data.getint(
                section_name, "number_of_processes"
            )

    def sample(
        self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator
    ):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur
        us_array = np.zeros(
            (self.number_of_trajectories, control_dim, control_horizon - 1)
        )
        costs_array = np.zeros((self.number_of_trajectories, 1, 1))
        trajectories_list = []

        noises_queue = mp.JoinableQueue()
        results = mp.Queue()
        for i in range(self.number_of_trajectories):
            noises_queue.put(
                self.noise_sampler.sample(control_dim, control_horizon - 1)
            )
        for i in range(self.number_of_processes):
            p = mp.Process(
                target=self.sample_single_traj,
                args=(
                    state_start,
                    dynamics,
                    cost_evaluator,
                    v,
                    control_horizon,
                    noises_queue,
                    results,
                    i,
                ),
            )
            p.start()
        noises_queue.join()
        for i in range(self.number_of_trajectories):
            result = results.get()
            trajectories_list.append(result[0])
            us_array[i, :, :] = result[1]
            costs_array[i, 0, 0] = result[2]

        return trajectories_list, us_array, costs_array

    def sample_single_traj(
        self,
        state_start,
        dynamics,
        cost_evaluator,
        v,
        control_horizon,
        noises_queue,
        results,
        i,
    ):
        while noises_queue.empty() is False:
            noises = noises_queue.get()
            state_cur = state_start
            trajectory = np.zeros((dynamics.get_state_dim()[0], control_horizon))
            trajectory[:, 0] = state_cur
            if (
                i
                > (1 - self.uncontrolled_trajectories_portion)
                * self.number_of_trajectories
            ):
                u = v + noises
            else:
                u = noises
            cost = 0
            for j in range(control_horizon - 1):
                cost += cost_evaluator.evaluate(
                    state_cur.reshape((-1, 1)), u[:, j : j + 1], dynamics=dynamics
                )
                state_cur = dynamics.propagate(state_cur, u[:, j])
                trajectory[:, j + 1] = state_cur
            cost += cost_evaluator.evaluate_terminal_cost(
                state_cur.reshape((-1, 1)), dynamics=dynamics
            )
            results.put([trajectory, u, cost])
            noises_queue.task_done()


class CCMPPIStochasticTrajectoriesSampler(StochasticTrajectoriesSampler):
    def __init__(
        self,
        number_of_trajectories=None,
        uncontrolled_trajectories_portion=None,
        noise_sampler=None,
    ):
        StochasticTrajectoriesSampler.__init__(
            self,
            number_of_trajectories,
            uncontrolled_trajectories_portion,
            noise_sampler,
        )

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(
            self, config_data, section_name
        )
        covariance_steering_helper_section_name = config_data.get(
            section_name, "covariance_steering_helper"
        )
        self.covariance_steering_helper = factory_from_config(
            covariance_steering_helper_factory_base,
            config_data,
            covariance_steering_helper_section_name,
        )

    def sample(
        self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator
    ):

        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur.copy()
        costs = np.zeros((self.number_of_trajectories, 1, 1))
        state_cur = np.tile(
            state_start.reshape((-1, 1)), (1, self.number_of_trajectories)
        )
        trajectories = np.zeros(
            (self.number_of_trajectories, state_cur.shape[0], control_horizon)
        )
        trajectories[:, :, 0] = np.swapaxes(state_cur, 0, 1)
        noises = self.noise_sampler.sample(
            control_dim, (control_horizon - 1) * self.number_of_trajectories
        )
        noises = noises.reshape(
            (control_dim, (control_horizon - 1), self.number_of_trajectories)
        )
        num_controlled_trajectories = int(
            (1 - self.uncontrolled_trajectories_portion) * self.number_of_trajectories
        )
        us = np.zeros((v.shape[0], v.shape[1], self.number_of_trajectories))
        us[:, :, :num_controlled_trajectories] = np.expand_dims(v, axis=2)
        us += noises

        self.covariance_steering_helper.dynamics_linearizer.set_dynamics(dynamics)
        reference_trajectory = self.rollout_out(state_start, v, dynamics)
        Ks, As, Bs, _, _, _ = self.covariance_steering_helper.covariance_control(
            state=state_cur.T,
            ref_state_vec=reference_trajectory.T,
            ref_ctrl_vec=v.T,
            return_sx=True,
            Sigma_epsilon=self.noise_sampler.covariance,
        )
        y = np.zeros((num_controlled_trajectories, state_start.shape[0], 1))
        for j in range(control_horizon - 1):
            K_aug = np.tile(Ks[j, :, :], (num_controlled_trajectories, 1, 1))
            us[:, j : j + 1, :num_controlled_trajectories] = us[
                :, j : j + 1, :num_controlled_trajectories
            ] + (K_aug @ y).reshape(
                us[:, j : j + 1, :num_controlled_trajectories].shape
            )
            A_aug = np.tile(As[j, :, :], (num_controlled_trajectories, 1, 1))
            B_aug = np.tile(Bs[j, :, :], (num_controlled_trajectories, 1, 1))
            noises_aug = noises[:, j, :num_controlled_trajectories].reshape(
                (
                    num_controlled_trajectories,
                    noises[:, j, :num_controlled_trajectories].shape[0],
                    1,
                )
            )
            y = A_aug @ y + B_aug @ noises_aug

            costs += cost_evaluator.evaluate(
                state_cur, us[:, j, :], noises[:, j, :], dynamics=dynamics
            )
            state_cur = dynamics.propagate(state_cur, us[:, j, :])
            trajectories[:, :, j + 1] = np.swapaxes(state_cur, 0, 1)
        costs += cost_evaluator.evaluate_terminal_cost(state_cur, dynamics=dynamics)
        us = np.moveaxis(us, 2, 0)
        return trajectories, us, costs

    def rollout_out(self, state_cur, v, dynamics):
        trajectory = np.zeros((dynamics.get_state_dim()[0], v.shape[1] + 1))
        trajectory[:, 0] = state_cur
        for i in range(v.shape[1]):
            state_next = dynamics.propagate(state_cur, v[:, i])
            trajectory[:, i + 1] = state_next
            state_cur = state_next
        return trajectory


class CCMPPIStochasticTrajectoriesSamplerSLowLoop(StochasticTrajectoriesSampler):
    def __init__(
        self,
        number_of_trajectories=None,
        uncontrolled_trajectories_portion=None,
        noise_sampler=None,
    ):
        StochasticTrajectoriesSampler.__init__(
            self,
            number_of_trajectories,
            uncontrolled_trajectories_portion,
            noise_sampler,
        )

    def initialize_from_config(self, config_data, section_name):
        StochasticTrajectoriesSampler.initialize_from_config(
            self, config_data, section_name
        )
        covariance_steering_helper_section_name = config_data.get(
            section_name, "covariance_steering_helper"
        )
        self.covariance_steering_helper = factory_from_config(
            covariance_steering_helper_factory_base,
            config_data,
            covariance_steering_helper_section_name,
        )

    def sample(
        self, state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator
    ):
        #  state_cur is the current state, v is the nominal control sequence
        state_start = state_cur.reshape((-1, 1))
        us = np.zeros((self.number_of_trajectories, control_dim, control_horizon - 1))
        costs = np.zeros((self.number_of_trajectories, 1, 1))
        trajectories = []
        self.covariance_steering_helper.dynamics_linearizer.set_dynamics(dynamics)
        reference_trajectory = self.rollout_out(state_cur, v, dynamics)
        (
            Ks,
            As,
            Bs,
            ds,
            Sx_cc,
            Sx_nocc,
        ) = self.covariance_steering_helper.covariance_control(
            state=state_cur.T,
            ref_state_vec=reference_trajectory.T,
            ref_ctrl_vec=v.T,
            return_sx=True,
            Sigma_epsilon=self.noise_sampler.covariance,
        )
        for i in range(self.number_of_trajectories):
            state_cur = state_start
            trajectory = np.zeros((dynamics.get_state_dim()[0], control_horizon))
            trajectory[:, :1] = state_cur
            noises = self.noise_sampler.sample(control_dim, control_horizon - 1)
            if (
                i
                > self.uncontrolled_trajectories_portion * self.number_of_trajectories
                - 0.001
            ):
                u = v + noises
            else:
                u = noises
            cost = 0
            y = np.zeros(state_cur.shape)
            for j in range(control_horizon - 1):
                if (
                    i
                    > self.uncontrolled_trajectories_portion
                    * self.number_of_trajectories
                    - 0.001
                ):
                    u[:, j : j + 1] = u[:, j : j + 1] + np.dot(Ks[j, :, :], y)
                y = np.dot(As[j, :, :], y) + np.dot(Bs[j, :, :], noises[:, j]).reshape(
                    (-1, 1)
                )
                cost += cost_evaluator.evaluate(
                    state_cur, u[:, j : j + 1], dynamics=dynamics
                )
                state_cur = dynamics.propagate(state_cur, u[:, j]).reshape((-1, 1))
                trajectory[:, j + 1 : j + 2] = state_cur
            cost += cost_evaluator.evaluate(
                state_cur, dynamics=dynamics
            )  # final cost TODO: add final_cost_evaluate() function to cost_evaluator
            trajectories.append(trajectory)
            us[i, :, :] = u
            costs[i, 0, 0] = cost
        return trajectories, us, costs

    def rollout_out(self, state_cur, v, dynamics):
        trajectory = np.zeros((dynamics.get_state_dim()[0], v.shape[1] + 1))
        trajectory[:, 0] = state_cur
        for i in range(v.shape[1]):
            state_next = dynamics.propagate(state_cur, v[:, i])
            trajectory[:, i + 1] = state_next
            state_cur = state_next
        return trajectory
