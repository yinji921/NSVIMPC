import functools as ft
from typing import NamedTuple

import einops as ei
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from attrs import define
from flax import struct
from loguru import logger
from og.cfg_utils import Cfg
from og.dyn_types import BBHFloat, BHFloat, BObs, HFloat, Obs, TBool, THFloat, TObs
from og.grad_utils import compute_norm_and_clip
from og.jax_types import BBFloat, FloatDict
from og.jax_utils import jax_vmap, merge01
from og.networks.network_utils import ActLiteral, HidSizes, get_act_from_str
from og.networks.optim import get_default_tx
from og.rng import PRNGKey
from og.schedules import Schedule, as_schedule
from og.train_state import TrainState
from og.tree_utils import tree_copy, tree_split_dims

from ncbf.avoid_utils import get_max_gae_term, get_max_mc
from ncbf.drone_task import get_h_vector_drone, state_to_obs_drone
from ncbf.networks.mlp import MLP
from ncbf.networks.value_net import ValueNet


@define
class TrainOfflineCfg(Cfg):
    act: ActLiteral
    Vh_act: ActLiteral
    hids: HidSizes

    lr: Schedule
    wd: Schedule

    n_batches: int

    disc_gamma: float
    gae_lambda: float

    # <- old + step * (new - old). 0 is no update, 1 is full update.
    ema_step: float

    clip_grad: float = 1.0


class Traj(NamedTuple):
    Tp1_obs: TObs
    Th_h: THFloat
    T_isterm: TBool

    @staticmethod
    def concatenate(trajs: list["Traj"]) -> "Traj":
        bTp1_obs = np.concatenate([traj.Tp1_obs for traj in trajs], axis=0)
        bTh_h = np.concatenate([traj.Th_h for traj in trajs], axis=0)
        bT_isterm = np.concatenate([traj.T_isterm for traj in trajs], axis=0)
        return Traj(bTp1_obs, bTh_h, bT_isterm)


class TrainOfflineDroneAlg(struct.PyTreeNode):
    Cfg = TrainOfflineCfg

    # How many times update() has been called.
    update_idx: int

    key: PRNGKey
    value_net: TrainState[HFloat]
    ema: dict

    obs_mean: Obs
    obs_std: Obs
    cfg: TrainOfflineCfg = struct.field(pytree_node=False)

    class Batch(NamedTuple):
        b_obs: BObs
        bh_h: BHFloat
        bh_Qh: BHFloat

        @property
        def batch_size(self):
            return len(self.b_obs)

    class EvalData(NamedTuple):
        bb_pos: BBFloat
        bbh_Vh: BBHFloat

        # Check against a traj.
        Th_h: THFloat
        Th_h_disc: THFloat
        Th_Qh_gae: THFloat
        Th_Vh_eval: THFloat
        Th_Vh_eval_ema: THFloat

        info: dict

    @classmethod
    def create(cls, key: PRNGKey, obs_mean, obs_std, nh: int, cfg: TrainOfflineCfg):
        key, key_quantile = jr.split(key, 2)
        act = get_act_from_str(cfg.act)
        Vh_act = get_act_from_str(cfg.Vh_act)

        value_net_cls = ft.partial(MLP, cfg.hids, act)
        value_net_def = ValueNet(value_net_cls, nh, Vh_act)
        value_net_tx = get_default_tx(as_schedule(cfg.lr).make(), as_schedule(cfg.wd).make())
        value_net = TrainState.create_from_def(key_quantile, value_net_def, (obs_mean,), value_net_tx)
        ema = tree_copy(value_net.params)

        zero = jnp.array(0, dtype=jnp.int32)
        return TrainOfflineDroneAlg(zero, key, value_net, ema, obs_mean, obs_std, cfg)

    def _compute_targets(self, b_traj: Traj) -> "TrainOfflineDroneAlg.Batch":
        batch_size, T, nh = b_traj.Th_h.shape

        # Compute predicted values.
        bTp1h_Vh = jax_vmap(self.value_net.apply, rep=2)(b_traj.Tp1_obs)

        # max_gae_fn = ft.partial(get_max_gae, self.cfg.disc_gamma, self.cfg.gae_lambda)
        max_gae_fn = ft.partial(get_max_gae_term, self.cfg.disc_gamma, self.cfg.gae_lambda)
        bTh_Qh = jax_vmap(max_gae_fn)(b_traj.Th_h, bTp1h_Vh, b_traj.Th_h, b_traj.T_isterm)
        # 3: Make the dataset by flattening (b, T) -> (b * T,)
        bT_obs = b_traj.Tp1_obs[:, :-1]
        bT_batch = TrainOfflineDroneAlg.Batch(bT_obs, b_traj.Th_h, bTh_Qh)
        b_batch = jax.tree_map(merge01, bT_batch)
        #  gae and observation values for each state, not entire trajectories so we can just merge them into a single array.
        # ipdb.set_trace()
        return b_batch

    @ft.partial(jax.jit, donate_argnums=(0,))
    def update(self, b_traj: Traj):
        # 1: Get dset.
        b_dset = self._compute_targets(b_traj)

        n_batches = self.cfg.n_batches
        assert b_dset.batch_size % n_batches == 0
        batch_size = b_dset.batch_size // self.cfg.n_batches
        logger.info(f"Using {n_batches} minibatches each epoch!")

        # 2: Shuffle and reshape
        key_shuffle, key_self = jr.split(self.key, 2)
        rand_idxs = jr.permutation(key_shuffle, jnp.arange(b_dset.batch_size))
        b_dset = jax.tree_map(lambda x: x[rand_idxs], b_dset)
        mb_dset = tree_split_dims(b_dset, (n_batches, batch_size))
        # ipdb.set_trace()

        # 3: Perform value function and policy updates.
        def updates_body(alg_: TrainOfflineDroneAlg, b_batch: TrainOfflineDroneAlg.Batch):
            return alg_._update_value(b_batch)

        new_self, info = lax.scan(updates_body, self, mb_dset, length=n_batches)
        # Take the mean.
        info = jax.tree_map(jnp.mean, info)

        return new_self.replace(key=key_self, update_idx=self.update_idx + 1), info

    def _update_value(self, batch: Batch) -> tuple["TrainOfflineDroneAlg", FloatDict]:
        def get_Vh_loss(params):
            bh_Vh_resid = jax.vmap(ft.partial(self.value_net.apply_with, params=params))(batch.b_obs)
            # bh_Vh = batch.bh_h + bh_Vh_resid
            bh_Vh = bh_Vh_resid
            loss_Vh = jnp.mean((bh_Vh - batch.bh_Qh) ** 2)
            # ipdb.set_trace()
            info = {
                "loss": loss_Vh,
            }
            return loss_Vh, info

        grads_Vh, Vh_info = jax.grad(get_Vh_loss, has_aux=True)(self.value_net.params)
        grads_Vh, Vh_info["grad"] = compute_norm_and_clip(grads_Vh, self.cfg.clip_grad)

        value_net = self.value_net.apply_gradients(grads=grads_Vh)

        # Apply ema.
        ema = optax.incremental_update(value_net.params, self.ema, self.cfg.ema_step)

        return self.replace(value_net=value_net, ema=ema), Vh_info

    def get_ema(self, obs):
        return self.value_net.apply_with(obs, params=self.ema)

    def get_Vh(self, obs, h, params=None):
        if params is None:
            params = self.value_net.params
        resid = self.value_net.apply_with(obs, params=params)
        # return h + resid
        return resid

    def get_Vh_ema(self, obs, h):
        return self.get_Vh(obs, h, params=self.ema)

    @ft.partial(jax.jit)
    def eval(self, T_obs_eval: TObs, Th_h_eval: THFloat) -> EvalData:
        # Visualize the learned value function.
        # [ px, py, theta, vx, vy, omega ]
        x0 = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0])

        n_px = 560
        n_py = 50

        with jax.ensure_compile_time_eval():
            b_x = np.linspace(-4.0, 10.0, num=n_px)
            b_y = np.linspace(0.0, 1.25, num=n_py)

        bb_x, bb_y = jnp.meshgrid(b_x, b_y)

        b1, b2 = bb_x.shape
        assert (b1, b2) == (n_py, n_px)

        bb_state = ei.repeat(x0, "nx -> b1 b2 nx", b1=b1, b2=b2)
        bb_state = jnp.array(bb_state)
        bb_state = bb_state.at[..., 0].set(bb_x)
        bb_state = bb_state.at[..., 1].set(bb_y)

        bb_pos = bb_state[..., :2]

        bb_obs = jax_vmap(state_to_obs_drone, rep=2)(bb_state)
        bbh_h = jax_vmap(get_h_vector_drone, rep=2)(bb_state)

        bb_obs_norm = (bb_obs - self.obs_mean) / self.obs_std
        bbh_Vh_resid = self.value_net.apply_with(bb_obs_norm, params=self.ema)
        # bbh_Vh = bbh_h + bbh_Vh_resid
        bbh_Vh = bbh_Vh_resid

        ##########################################################################
        # Compute the (discounted) value function empirically.
        Th_h_disc_eval = get_max_mc(self.cfg.disc_gamma, Th_h_eval, Th_h_eval)
        # Evaluate the predicted value function.
        Th_Vh_eval = jax_vmap(self.get_Vh)(T_obs_eval, Th_h_eval)

        # Model what we are doing, i.e., dummy observation.
        Tp1h_Vh_eval = jnp.concatenate([Th_Vh_eval, 1e8 * Th_Vh_eval[-1:]], axis=0)

        T_isterm = np.zeros(len(Th_Vh_eval), dtype=bool)
        T_isterm[-1] = True

        # Compute the GAE estimate of the value function target
        # (i.e., interpolated version between Th_h_diisc_eval and Th_Vh_eval)
        max_gae_fn = ft.partial(get_max_gae_term, self.cfg.disc_gamma, self.cfg.gae_lambda)
        Th_Qh_gae = max_gae_fn(Th_h_eval, Tp1h_Vh_eval, Th_h_eval, T_isterm)

        # Evaluate a smoothed (EMA) version of the predicted value function.
        Th_Vh_eval_ema = jax_vmap(self.get_Vh_ema)(T_obs_eval, Th_h_eval)
        ##########################################################################

        info = {
            "Vh_evaltraj_err": jnp.mean((Th_Vh_eval - Th_h_disc_eval) ** 2),
        }

        return self.EvalData(bb_pos, bbh_Vh, Th_h_eval, Th_h_disc_eval, Th_Qh_gae, Th_Vh_eval, Th_Vh_eval_ema, info)
