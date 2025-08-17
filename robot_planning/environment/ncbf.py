import functools as ft
import pathlib
from typing import NamedTuple

import einops as ei
import ipdb
import jax
import jax.debug as jd
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import loguru
import numpy as np
from og.ckpt_utils import load_ckpt_ez
from og.dyn_types import HFloat
from og.normalization import MeanStd
from og.schedules import Constant
from og.train_state import TrainState

from ncbf.ar_task import ConstrCfg, ObsCfg, get_h_vector, state_to_obs
from ncbf.offline.train_offline_alg import TrainOfflineAlg, TrainOfflineCfg
from ncbf.scripts.ncbf_config import get_cfgs, get_h_cfg_for
from robot_planning.controllers.MPPI.stochastic_trajectories_sampler import MPPIStochasticTrajectoriesSampler
from robot_planning.environment.cost_evaluators import AutorallyMPPICostEvaluator
from robot_planning.environment.dynamics.autorally_dynamics.autorally_dynamics import AutoRallyDynamics
from robot_planning.helper.path_utils import get_commit_ckpt_dir, get_runs_dir


class SamplerResult(NamedTuple):
    bxT_state: jnp.ndarray
    buT_control: jnp.ndarray
    b11_cost: jnp.ndarray


class DetailedSamplerResult(NamedTuple):
    result: SamplerResult
    info: dict


class NCBF:
    def __init__(self, ckpt_path: pathlib.Path | None = None):
        # n_kappa = 10
        # self.obs_cfg = ObsCfg(kappa_dt=0.2, n_kappa=n_kappa, dt_exp=1.25)
        # self.h_cfg = ConstrCfg(track_width=1.3, track_width_term=1.9, margin_lo=0.2, margin_hi=1.0, h_term=2.2)

        # Change the track_width in get_cfgs() function.
        self.obs_cfg, self.h_cfg = get_cfgs()
        self.value_net, self.obs_norm, self.alg_cfg, self.nh = self.load_value_net(ckpt_path)

    def load_value_net(
        self, ckpt_path: pathlib.Path | None = None
    ) -> tuple[TrainState[HFloat], MeanStd, TrainOfflineCfg, int]:
        hids = [256, 256]

        if ckpt_path is None:
            # ckpt_path = get_commit_ckpt_dir() / "0041-maple-pastr/00001000/default"
            # ckpt_path = get_commit_ckpt_dir() / "0041-maple-pastr/00050000/default"
            # ckpt_path = get_commit_ckpt_dir() / "0041-maple-pastr/00099999/default"
            # ckpt_path = get_commit_ckpt_dir() / "0026-jumpi-sound/ckpts/00099000/default" # counter_clockwise, disturbed, using MPPI data
            ckpt_path = get_commit_ckpt_dir() / "0025-true-haze/ckpts/00099000/default" # counter_clockwise, disturbed, using Shield-MPPI data
            # ckpt_path = get_commit_ckpt_dir() / "0023-comfy-shado/ckpts/00099000/default" # counter_clockwise, using MPPI data"
            # ckpt_path = get_commit_ckpt_dir() / "0024-dandy-lake/ckpts/00099000/default" # counter_clockwise, using Shield-MPPI data
            # ckpt_path = get_commit_ckpt_dir() / "0076-misun-glade/ckpts/00040000/default" # counter_clockwise cluttered env, using MPPI data

            # # This one uses track_width = 1.3.
            # ckpt_path = get_commit_ckpt_dir() / "0057-track1.3_v9_horig/00200000/default" # counter_clockwise, using MPPI data, width 1.3
            # self.h_cfg = get_h_cfg_for(width=1.3)

            # This one uses track_width = 1.2
            # ckpt_path = get_commit_ckpt_dir() / "0058-track1.2_v9_horig/00100000/default" # counter_clockwise, using MPPI data, width 1.2
            # self.h_cfg = get_h_cfg_for(width=1.2)

            # # This one uses track_width = 1.2
            # # Sometime can have some weird behaviors
            # ckpt_path = get_commit_ckpt_dir() / "0058-track1.2_v9_horig/00200000/default" # counter_clockwise, using MPPI data, width 1.2
            # self.h_cfg = get_h_cfg_for(width=1.2)

            # Another one using track_width = 1.2
            # ckpt_path = (
            #     get_commit_ckpt_dir() / "0060-track1.3_v8_horig/00299999/default"
            # )  # counter_clockwise, using MPPI data, width 1.2, nominal policy uses vel_tgt=8
            # self.h_cfg = get_h_cfg_for(width=1.2)

            # Track_width = 1.1
            # ckpt_path = (
            #         get_commit_ckpt_dir() / "0063-track1.1_v50_horig_bigdset/00100000/default"
            # )  # counter_clockwise, using MPPI data, width 1.1, nominal policy uses vel_tgt=50
            # ckpt_path = (
            #         get_commit_ckpt_dir() / "0063-track1.1_v50_horig_bigdset/00200000/default"
            # )  # counter_clockwise, using MPPI data, width 1.1, nominal policy uses vel_tgt=50
            # self.h_cfg = get_h_cfg_for(width=1.1)

            # Track_width = 1.1, gamma = 0.93
            # ckpt_path = (
            #         get_commit_ckpt_dir() / "0064-track1.1_v50_horig_bigdset_gam93/00040000/default"
            # )  # counter_clockwise, using MPPI data, width 1.1, nominal policy uses vel_tgt=50
            # self.h_cfg = get_h_cfg_for(width=1.1)

            # Track_width = 1.1, gamma = 0.94
            # ckpt_path = (
            #         get_commit_ckpt_dir() / "0065-track1.1_v50_horig_bigdset_gam94/00100000/default"
            # )  # counter_clockwise, using MPPI data, width 1.1, nominal policy uses vel_tgt=50
            # self.h_cfg = get_h_cfg_for(width=1.1)

            # # Track_width = 1.1, gamma = 0.94, smaller NN.
            # # counter_clockwise, using MPPI data, width 1.1, nominal policy uses vel_tgt=50
            # # ckpt_path = (
            # #         get_commit_ckpt_dir() / "0065-track1.1_v50_horig_bigdset_gam94/00100000/default"
            # # )  # counter_clockwise, using MPPI data, width 1.1, nominal policy uses vel_tgt=50
            # # ckpt_path = get_runs_dir() / "offline/0067-hid6464_2/ckpts/00020000/default"
            # ckpt_path = get_runs_dir() / "offline/0067-hid6464_2/ckpts/00070000/default"
            # self.h_cfg = get_h_cfg_for(width=1.1)
            # hids = [64, 64]

            # # Track_width = 1.1, gamma = 0.94, smaller NN.
            # # counter_clockwise, using MPPI data, width 1.1, nominal policy uses vel_tgt=50
            # # ckpt_path = (
            # #         get_commit_ckpt_dir() / "0065-track1.1_v50_horig_bigdset_gam94/00100000/default"
            # # )  # counter_clockwise, using MPPI data, width 1.1, nominal policy uses vel_tgt=50
            # # ckpt_path = get_runs_dir() / "offline/0069-hid9696/ckpts/00100000/default"
            # ckpt_path = get_runs_dir() / "0069-hid9696/00100000/default"
            # self.h_cfg = get_h_cfg_for(width=1.1)
            # hids = [96, 96]

        lr = Constant(3e-4)
        wd = Constant(1e-2)
        n_batches = 1
        disc_gamma = 0.92
        gae_lambda = 0.95
        ema_step = 1e-3
        cfg = TrainOfflineCfg("relu", "softplus", hids, lr, wd, n_batches, disc_gamma, gae_lambda, ema_step)

        # Load cfg.
        ckpt_dict = load_ckpt_ez(ckpt_path, {"cfg": cfg})
        cfg = TrainOfflineCfg.fromdict(ckpt_dict["cfg"])

        state_cartesian = np.zeros(8)
        curvilinear = np.zeros(3)
        nh = len(get_h_vector(self.h_cfg, state_cartesian, curvilinear))

        dummy = jnp.zeros(1)
        alg: TrainOfflineAlg = TrainOfflineAlg.create(jr.PRNGKey(0), dummy, dummy, nh, cfg)

        # Load ckpt.
        ckpt_dict = load_ckpt_ez(ckpt_path, {"alg": alg})
        alg = ckpt_dict["alg"]
        print("Loaded ckpt from {}! update steps={}".format(ckpt_path, alg.update_idx))

        # Replace the params with the ema params.
        value_net = alg.value_net.replace(params=alg.ema)
        value_net = value_net.strip()
        obs_norm = MeanStd(alg.obs_mean, alg.obs_std)

        return value_net, obs_norm, cfg, nh

    def get_norm_obs(self, state: jnp.ndarray, map_state: jnp.ndarray, dynamics: AutoRallyDynamics):
        assert state.shape == (8,)
        assert map_state.shape == (3,)
        track = dynamics.track
        obs = state_to_obs(track, self.obs_cfg, state, map_state)
        norm_obs = self.obs_norm.normalize(obs)
        return norm_obs

    def get_h_vector(self, state: jnp.ndarray, map_state: jnp.ndarray):
        assert state.shape == (8,)
        assert map_state.shape == (3,)
        h_h = get_h_vector(self.h_cfg, state, map_state)
        return h_h

    def get_h(
        self, state: jnp.ndarray, map_state: jnp.ndarray, dynamics: AutoRallyDynamics, ncbf_weights=None
    ) -> jnp.ndarray:
        # 1: Get the NORMALIZED observation.
        norm_obs = self.get_norm_obs(state, map_state, dynamics)

        # 2: Get the base h value.
        h_h = self.get_h_vector(state, map_state)

        # 3: Get the learned residual.
        if ncbf_weights is None:
            loguru.logger.info("Using the saved weights!")
            ncbf_weights = self.value_net.params
        else:
            loguru.logger.info("Using the passed in weights!")

        h_Vh_resid = self.value_net.apply_with(norm_obs, params=ncbf_weights)

        # return h_h + h_Vh_resid
        h_h_out = jnp.maximum(h_h, h_Vh_resid)

        # HACK: If the velocity is too low, then just use h_h.
        vx, vy = state[0], state[1]
        low_vel = jnp.abs(vx) <= 4.0
        h_h_out = jnp.where(low_vel, h_h, h_h_out)

        return h_h_out


class AutorallyMPPINCBFCostEvaluator(AutorallyMPPICostEvaluator):
    def __init__(
        self,
        goal_checker=None,
        collision_checker=None,
        Q=None,
        R=None,
        collision_cost=None,
        goal_cost=None,
    ):
        AutorallyMPPICostEvaluator.__init__(self, goal_checker, collision_checker, Q, R, collision_cost, goal_cost)
        self.ncbf = NCBF()
        self.cbf_alpha = 0.9
        self.cbf_alpha_resample = 0.5
        # self.cbf_alpha_resample = 0.85
        # self.cbf_vio_cost = 1_000.0
        self.cbf_vio_cost = 2_000.0
        self.cbf_safe_cost = 100.0

        self.track_width = 1.5

    def initialize_from_config(self, config_data, section_name: str):
        # Get superclass parameters
        AutorallyMPPICostEvaluator.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name,"include_cbf_cost"):
            self.include_cbf_cost = config_data.getboolean(section_name, "include_cbf_cost")
        else:
            self.include_cbf_cost = True

    def evaluate_cost(self, state_cur, state_next, action, noise, dynamics: AutoRallyDynamics, ncbf_weights=None):
        """
        :param state_cur: (nx, ) = (8, )
        :param state_next: (nx, ) = (8, )
        :param action: (nu, ) = (2, )
        :param noise: (nu, ) = (2, )
        :param dynamics:
        :param ncbf_weights: Weights for the NCBF to use instead of the saved ones..
        :return:
        """
        assert action is not None
        assert noise is not None

        assert state_cur.shape == state_next.shape
        b_state = jnp.stack([state_cur, state_next], axis=0)

        to_local = ft.partial(self.global_to_local_coordinate_transform, dynamics=dynamics)

        # 1: State cost
        #       This function assumes a batch dimension at the end.
        b_map_state = jax.vmap(to_local)(b_state[:, :, None]).squeeze(-1)
        assert b_map_state.shape == (2, 8)

        map_state = b_map_state[0]
        err_state = map_state - self.goal_checker.goal_state
        assert err_state.shape == (8,)
        cost_state = 0.5 * jnp.dot(err_state, jnp.dot(self.Q, err_state))

        # 2: Control cost
        #       1/2 eRe
        cost_noise1 = 0.5 * jnp.dot(noise, jnp.dot(self.R, noise))
        cost_noise2 = jnp.dot(noise, jnp.dot(self.R, action - noise))
        cost_control = 0.5 * jnp.dot(action, jnp.dot(self.R, action))
        cost_controls = cost_noise1 + cost_noise2 + cost_control

        # 3: Collision cost.
        #       Again, this assumes a batch dimension at the end.
        collisions = self.collision_checker.check(map_state[:, None], state_cur[:, None]).squeeze(-1)
        assert collisions.shape == tuple()
        cost_collision = jnp.where(collisions, self.collision_cost, 0.0)

        # 4: CBF cost.
        get_h = ft.partial(self.ncbf.get_h, dynamics=dynamics, ncbf_weights=ncbf_weights)
        bh_Vh = jax.vmap(get_h)(b_state[:, :8], b_map_state[:, -3:])
        h_Vh_now, h_Vh_next = bh_Vh[0], bh_Vh[1]

        h_h_now = self.ncbf.get_h_vector(state_cur, b_map_state[0, -3:])
        h_h_next = self.ncbf.get_h_vector(state_next, b_map_state[1, -3:])

        #   Compute the violation of the discrete time CBF condition.
        #   \dot{V} + λ V <= 0.
        #   =>  V(x_{t + Δ}) <= exp(-λ Δ) V(x_t).
        #   => V(x_{t + Δ}) - exp(-λ Δ) V(x_t) <= 0.
        #   => V(x_{t + Δ}) - α V(x_t) <= 0.
        #   λ=0 => α=1,    λ=∞ => α=0
        #   α=1 is the most conservative, α=0 is the least conservative.
        h_vio = h_Vh_next - self.cbf_alpha * h_Vh_now
        h_cost_cbf = jnp.where(h_vio > 0, self.cbf_vio_cost * h_vio, self.cbf_safe_cost * h_vio)
        # #       If we are unsafe now, then maximize safety of the next step.
        # is_unsafe = jnp.any(h_Vh_now > 0)
        # h_cost_cbf_unsafe = self.cbf_vio_cost * h_Vh_next
        # h_cost_cbf = jnp.where(is_unsafe, h_cost_cbf_unsafe, h_cost_cbf)
        cost_cbf = jnp.sum(h_cost_cbf)

        # Sum all costs, reshape and return.
        if self.include_cbf_cost:
            cost = cost_state + cost_controls + cost_collision + cost_cbf
        else:
            print("not including cbf cost")
            cost = cost_state + cost_controls + cost_collision
        assert cost.shape == tuple()

        Vh_unsafe_thresh = -0.05
        next_unsafe_Vh = jnp.any(h_Vh_next > Vh_unsafe_thresh)
        # h_vio_resample = h_Vh_now - self.cbf_alpha_resample * h_Vh_next
        h_vio_resample = h_h_next - self.cbf_alpha_resample * h_h_now
        # h_vio_resample = h_h_now - self.cbf_alpha_resample * h_h_next # TODO: This should be h_h_next - self.cbf_alpha_resample * h_h_now

        violate_cbf_resample = jnp.max(h_vio_resample) > -0.05
        is_unsafe = next_unsafe_Vh | violate_cbf_resample | collisions

        # is_unsafe = next_unsafe_Vh | collisions
        # is_unsafe = jnp.any(h_vio > 0) | collisions
        # is_unsafe = collisions

        # info = {"h_vio": h_vio, "h_Vh_next": h_Vh_next, "h_h_next": h_h_next}
        info = {"h_vio": h_vio, "h_h_curr": h_h_now, "h_h_next": h_h_next, "h_Vh_curr": h_Vh_now, "h_Vh_next": h_Vh_next}

        return cost[None, None], is_unsafe, info

    def evaluate_cbf_cost(self, state_cur, dynamics, state_next):
        ncbf_weights = self.ncbf.value_net.params

        print("Tracing ncbf cbf cost...", end="")
        assert state_cur.shape == state_next.shape == (8, 1)
        state_cur = state_cur.squeeze()
        state_next = state_next.squeeze()
        assert state_cur.shape == state_next.shape == (8,)

        b_state = jnp.stack([state_cur, state_next], axis=0)

        to_local = ft.partial(self.global_to_local_coordinate_transform, dynamics=dynamics)

        b_map_state = jax.vmap(to_local)(b_state[:, :, None]).squeeze(-1)
        assert b_map_state.shape == (2, 8)

        USE_NCBF_FOR_REPAIR = False
        if USE_NCBF_FOR_REPAIR:
            get_h = ft.partial(self.ncbf.get_h, dynamics=dynamics, ncbf_weights=ncbf_weights)
            bh_Vh = jax.vmap(get_h)(b_state[:, :8], b_map_state[:, -3:])
            h_Vh_now, h_Vh_next = bh_Vh[0], bh_Vh[1]

            #   Compute the violation of the discrete time CBF condition.
            #   \dot{V} + λ V <= 0.
            #   =>  V(x_{t + Δ}) <= exp(-λ Δ) V(x_t).
            #   => V(x_{t + Δ}) - exp(-λ Δ) V(x_t) <= 0.
            #   => V(x_{t + Δ}) - α V(x_t) <= 0.
            #   λ=0 => α=1,    λ=∞ => α=0
            #   α=1 is the most conservative, α=0 is the least conservative.
            h_vio = h_Vh_next - self.cbf_alpha * h_Vh_now
            h_vio = jnp.maximum(h_vio, -0.1)
            vio = h_vio.sum()
        else:
            # Use handcrafted CBF for repair.
            track_width = self.track_width
            # logger.info("Using track_width {} for repair".format(track_width))

            def danger_index(x):
                return x[-2] ** 2 - track_width**2

            b_Vh = jax.vmap(danger_index)(b_map_state)
            assert b_Vh.shape == (2,)
            Vh_now, Vh_next = b_Vh[0], b_Vh[1]

            hvio = Vh_next - self.cbf_alpha * Vh_now
            vio = jnp.maximum(hvio, -0.1)

        if self.collision_cost is not None:
            cost = vio * self.collision_cost
        else:
            cost = vio * 1000  # default collision cost

        print("Done")
        return cost

class MPPINCBFStochasticTrajectoriesSampler(MPPIStochasticTrajectoriesSampler):
    def __init__(self, *args, **kwargs):
        MPPIStochasticTrajectoriesSampler.__init__(self, *args, **kwargs)
        self.ncbf_weights = None

    def initialize_from_config(self, config_data, section_name):
        MPPIStochasticTrajectoriesSampler.initialize_from_config(self, config_data, section_name)

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
        cost_evaluator: AutorallyMPPINCBFCostEvaluator,
        control_bounds=None,
        opponent_agents=None,
    ):
        detailed_result = self._sample(
            state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator, control_bounds, self.ncbf_weights
        )
        return detailed_result.result

    def sample_detailed(
        self,
        state_cur,
        v,
        control_horizon: int,
        control_dim: int,
        dynamics: AutoRallyDynamics,
        cost_evaluator: AutorallyMPPINCBFCostEvaluator,
        control_bounds=None,
        opponent_agents=None,
    ) -> DetailedSamplerResult:
        detailed_result = self._sample(
            state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator, control_bounds, self.ncbf_weights
        )
        return detailed_result

    @ft.partial(jax.jit, static_argnames=("self", "control_horizon", "control_dim", "dynamics", "cost_evaluator"))
    def _sample(
        self,
        state_cur,
        v,
        control_horizon: int,
        control_dim: int,
        dynamics: AutoRallyDynamics,
        cost_evaluator: AutorallyMPPINCBFCostEvaluator,
        control_bounds=None,
        ncbf_weights=None,
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
        cost_fn = ft.partial(cost_evaluator.evaluate_cost, dynamics=dynamics, ncbf_weights=ncbf_weights)
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

        # Tbh_vio = info["h_vio"]
        # jd.print("")
        # for kk in range(1, 3):
        #     b_is_safe = jnp.all(jnp.all(Tbh_vio[:kk] < 0, axis=2), axis=0)
        #     n_safe = jnp.sum(b_is_safe)
        #     jd.print("Safe till {}: {}", kk, n_safe)
        # b_is_safe_all = jnp.all(jnp.all(Tbh_vio < 0, axis=2), axis=0)
        # n_safe_all = jnp.sum(b_is_safe_all)
        # jd.print("Safe till end: {}", n_safe_all)

        result = SamplerResult(bxT_state, buT_control, b11_cost)
        return DetailedSamplerResult(result, info)

class MPPINCBFStochasticTrajectoriesSamplerInefficient(MPPINCBFStochasticTrajectoriesSampler):
    def __init__(self, *args, **kwargs):
        MPPINCBFStochasticTrajectoriesSampler.__init__(self, *args, **kwargs)

    def sample(
        self,
        state_cur,
        v,
        control_horizon: int,
        control_dim: int,
        dynamics: AutoRallyDynamics,
        cost_evaluator: AutorallyMPPINCBFCostEvaluator,
        control_bounds=None,
        opponent_agents=None,
    ):
        detailed_result = self._sample(
            state_cur, v, control_horizon, control_dim, dynamics, cost_evaluator, control_bounds, self.ncbf_weights
        )
        return detailed_result.result

    @ft.partial(jax.jit, static_argnames=("self", "control_horizon", "control_dim", "dynamics", "cost_evaluator"))
    def _sample(
            self,
            state_cur,
            v,
            control_horizon: int,
            control_dim: int,
            dynamics: AutoRallyDynamics,
            cost_evaluator: AutorallyMPPINCBFCostEvaluator,
            control_bounds=None,
            ncbf_weights=None,
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
        cost_fn = ft.partial(cost_evaluator.evaluate_cost, dynamics=dynamics, ncbf_weights=ncbf_weights)
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