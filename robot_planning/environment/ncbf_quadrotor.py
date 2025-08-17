import functools as ft
import pathlib

import jax
import jax.numpy as jnp
import jax.random as jr
import loguru
import numpy as np
from og.ckpt_utils import load_ckpt_ez
from og.normalization import MeanStd
from og.schedules import Constant

from ncbf.drone_task import get_h_vector_drone, state_to_obs_drone
from ncbf.offline.train_offline_alg_drone import TrainOfflineCfg, TrainOfflineDroneAlg
from robot_planning.environment.cost_evaluators import AutorallyMPPICostEvaluator, QuadraticCostEvaluator
from robot_planning.environment.dynamics.autorally_dynamics.autorally_dynamics import AutoRallyDynamics
from robot_planning.helper.path_utils import get_drone_commit_ckpt_dir


class QuadrotorNCBF:
    def __init__(self, ckpt_path: pathlib.Path | None = None):
        self.value_net, self.obs_norm, self.alg_cfg, self.nh = self.load_value_net(ckpt_path)

    def load_value_net(self, ckpt_path: pathlib.Path | None = None):
        hids = [256, 256]

        if ckpt_path is None:
            # ckpt_path = get_drone_commit_ckpt_dir() / "0007-gamma88/ckpts/00299999/default"
            ckpt_path = get_drone_commit_ckpt_dir() / "0009-newdset_gamma85/ckpts/00299999/default"

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

        state = np.zeros(6)
        nh = len(get_h_vector_drone(state))

        dummy = jnp.zeros(1)
        alg: TrainOfflineDroneAlg = TrainOfflineDroneAlg.create(jr.PRNGKey(0), dummy, dummy, nh, cfg)

        # Load ckpt.
        ckpt_dict = load_ckpt_ez(ckpt_path, {"alg": alg})
        alg = ckpt_dict["alg"]
        print("Loaded ckpt from {}! update steps={}".format(ckpt_path, alg.update_idx))

        # Replace the params with the ema params.
        value_net = alg.value_net.replace(params=alg.ema)
        value_net = value_net.strip()
        obs_norm = MeanStd(alg.obs_mean, alg.obs_std)

        return value_net, obs_norm, cfg, nh

    def get_norm_obs(self, state: jnp.ndarray):
        assert state.shape == (6,)
        obs = state_to_obs_drone(state)
        norm_obs = self.obs_norm.normalize(obs)
        return norm_obs

    def get_h_vector(self, state: jnp.ndarray):
        assert state.shape == (6,)
        h_h = get_h_vector_drone(state)
        return h_h

    def get_h(self, state: jnp.ndarray, ncbf_weights=None) -> jnp.ndarray:
        # 1: Get the NORMALIZED observation.
        norm_obs = self.get_norm_obs(state)

        # 2: Get the base h value.
        h_h = self.get_h_vector(state)

        # 3: Get the learned residual.
        if ncbf_weights is None:
            loguru.logger.info("Using the saved weights!")
            ncbf_weights = self.value_net.params
        else:
            loguru.logger.info("Using the passed in weights!")

        h_Vh_resid = self.value_net.apply_with(norm_obs, params=ncbf_weights)
        h_h_out = jnp.maximum(h_h, h_Vh_resid)

        # # HACK: If the velocity is too low, then just use h_h.
        # vx, vy = state[3], state[4]
        # vel = jnp.sqrt(vx ** 2 + vy ** 2)
        # low_vel = vel <= 0.1
        # h_h_out = jnp.where(low_vel, h_h, h_h_out)

        return h_h_out


class QuadrotorMPPINCBFCostEvaluator(QuadraticCostEvaluator):
    def __init__(
        self,
        goal_checker=None,
        collision_checker=None,
        Q=None,
        R=None,
        collision_cost=None,
        goal_cost=None,
    ):
        QuadraticCostEvaluator.__init__(self, goal_checker, collision_checker, Q, R, collision_cost, goal_cost)
        self.ncbf = QuadrotorNCBF()
        self.cbf_alpha = 0.9
        self.cbf_alpha_localrepair = 0.1
        self.cbf_alpha_resample = 0.5
        self.cbf_alpha_resample_Vh = 0.9
        # self.cbf_alpha_resample = 0.85
        # self.cbf_vio_cost = 1_000.0
        self.cbf_vio_cost = 2_000.0
        self.cbf_safe_cost = 100.0

        self.Vh_unsafe_thresh = -0.05
        # self.Vh_unsafe_thresh = 0.5
        # self.Vh_unsafe_thresh = -0.1

        # self.Vh_unsafe_thresh = 0.0
        # self.Vh_unsafe_thresh = 0.2

        self.h_vio_resample_h_thresh = None
        # self.h_vio_resample_Vh_thresh = -0.05
        self.h_vio_resample_Vh_thresh = None

    def initialize_from_config(self, config_data, section_name: str):
        # Get superclass parameters
        AutorallyMPPICostEvaluator.initialize_from_config(self, config_data, section_name)
        if config_data.has_option(section_name, "include_cbf_cost"):
            self.include_cbf_cost = config_data.getboolean(section_name, "include_cbf_cost")
        else:
            self.include_cbf_cost = True

    def evaluate_cost(self, state_cur, state_next, action, noise, dynamics: AutoRallyDynamics, ncbf_weights=None):
        """
        :param state_cur: (nx, ) = (6, )
        :param state_next: (nx, ) = (6, )
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

        # 1: State cost
        #       This function assumes a batch dimension at the end.
        assert b_state.shape == (2, 6)

        err_state = state_cur - self.goal_checker.goal_state
        assert err_state.shape == (6,)
        cost_state = 0.5 * jnp.dot(err_state, jnp.dot(self.Q, err_state))

        # 2: Control cost
        #       1/2 eRe
        cost_noise1 = 0.5 * jnp.dot(noise, jnp.dot(self.R, noise))
        cost_noise2 = jnp.dot(noise, jnp.dot(self.R, action - noise))
        cost_control = 0.5 * jnp.dot(action, jnp.dot(self.R, action))
        cost_controls = cost_noise1 + cost_noise2 + cost_control

        # 3: Collision cost.
        #       Again, this assumes a batch dimension at the end.
        collisions = self.collision_checker.check(state_cur[:, None]).squeeze(-1)
        assert collisions.shape == tuple()
        cost_collision = jnp.where(collisions, self.collision_cost, 0.0)

        # 4: CBF cost.
        get_h = ft.partial(self.ncbf.get_h, ncbf_weights=ncbf_weights)
        bh_Vh = jax.vmap(get_h)(b_state)
        h_Vh_now, h_Vh_next = bh_Vh[0], bh_Vh[1]

        h_h_now = self.ncbf.get_h_vector(state_cur)
        h_h_next = self.ncbf.get_h_vector(state_next)

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

        Vh_unsafe_thresh = self.Vh_unsafe_thresh
        next_unsafe_Vh = jnp.any(h_Vh_next > Vh_unsafe_thresh)
        # h_vio_resample = h_Vh_now - self.cbf_alpha_resample * h_Vh_next

        h_vio_resample_Vh = h_Vh_next - self.cbf_alpha_resample * h_Vh_now
        h_vio_resample_h = h_h_next - self.cbf_alpha_resample_Vh * h_h_now

        # h_vio_resample = h_h_now - self.cbf_alpha_resample * h_h_next # TODO: This should be h_h_next - self.cbf_alpha_resample * h_h_now

        is_unsafe = next_unsafe_Vh | collisions

        if self.h_vio_resample_h_thresh is not None:
            vio_cbf_resample = jnp.max(h_vio_resample_h) > self.h_vio_resample_h_thresh
            is_unsafe = is_unsafe | vio_cbf_resample

        if self.h_vio_resample_Vh_thresh is not None:
            vio_cbf_resample_Vh = jnp.max(h_vio_resample_Vh) > self.h_vio_resample_Vh_thresh
            is_unsafe = is_unsafe | vio_cbf_resample_Vh

        # is_unsafe = next_unsafe_Vh | collisions
        # is_unsafe = jnp.any(h_vio > 0) | collisions
        # is_unsafe = collisions

        # info = {"h_vio": h_vio, "h_Vh_next": h_Vh_next, "h_h_next": h_h_next}
        info = {
            "h_vio": h_vio,
            "h_h_curr": h_h_now,
            "h_h_next": h_h_next,
            "h_Vh_curr": h_Vh_now,
            "h_Vh_next": h_Vh_next,
        }

        return cost[None, None], is_unsafe, info

    def evaluate_cbf_cost(self, state_cur, dynamics, state_next):
        ncbf_weights = self.ncbf.value_net.params

        print("Tracing ncbf cbf cost...", end="")
        assert state_cur.shape == state_next.shape == (6, 1)
        state_cur = state_cur.squeeze()
        state_next = state_next.squeeze()
        assert state_cur.shape == state_next.shape == (6,)

        b_state = jnp.stack([state_cur, state_next], axis=0)

        USE_NCBF_FOR_REPAIR = False
        if USE_NCBF_FOR_REPAIR:
            raise ValueError("")
            # get_h = ft.partial(self.ncbf.get_h, dynamics=dynamics, ncbf_weights=ncbf_weights)
            # bh_Vh = jax.vmap(get_h)(b_state[:, :8], b_map_state[:, -3:])
            # h_Vh_now, h_Vh_next = bh_Vh[0], bh_Vh[1]
            # #   Compute the violation of the discrete time CBF condition.
            # #   \dot{V} + λ V <= 0.
            # #   =>  V(x_{t + Δ}) <= exp(-λ Δ) V(x_t).
            # #   => V(x_{t + Δ}) - exp(-λ Δ) V(x_t) <= 0.
            # #   => V(x_{t + Δ}) - α V(x_t) <= 0.
            # #   λ=0 => α=1,    λ=∞ => α=0
            # #   α=1 is the most conservative, α=0 is the least conservative.
            # h_vio = h_Vh_next - self.cbf_alpha * h_Vh_now
            # h_vio = jnp.maximum(h_vio, -0.1)
            # vio = h_vio.sum()
        else:
            # Use handcrafted CBF for repair.
            def danger_index(x):
                h_h = get_h_vector_drone(x)
                return h_h

            bh_Vh = jax.vmap(danger_index)(b_state)
            nh = self.ncbf.nh
            assert bh_Vh.shape == (2, nh)
            h_Vh_now, h_Vh_next = bh_Vh[0], bh_Vh[1]

            h_hvio = h_Vh_next - self.cbf_alpha_localrepair * h_Vh_now
            h_vio = jnp.maximum(h_hvio, -0.1)
            vio = h_vio.sum()

        if self.collision_cost is not None:
            cost = vio * self.collision_cost
        else:
            cost = vio * 1000  # default collision cost

        print("Done")
        return cost
