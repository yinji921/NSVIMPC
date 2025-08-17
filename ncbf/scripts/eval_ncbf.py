import pathlib
import pickle

import ipdb
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from og.ckpt_utils import load_ckpt_ez
from og.jax_utils import jax_jit_np
from og.plot_utils import line_labels
from og.schedules import Constant

from ncbf.ar_task import ConstrCfg, ObsCfg, get_h_vector
from ncbf.avoid_utils import get_max_mc, get_max_gae_term
from ncbf.dset_offline import DsetOffline
from ncbf.offline.train_offline_alg import TrainOfflineAlg, TrainOfflineCfg
from ncbf.scripts.ncbf_config import get_cfgs
from robot_planning.helper.path_utils import get_root_dir
import functools as ft


def main(ckpt_path: pathlib.Path):
    # Load dset.
    data_dir = get_root_dir() / "data"
    dset_path = data_dir / "dset.pkl"

    ckpt_name = ckpt_path.parent.name

    with open(dset_path, "rb") as f:
        dset: DsetOffline = pickle.load(f)

    obs_cfg, h_cfg = get_cfgs()

    # Load ckpt.
    hids = [256, 256]
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
    nh = len(get_h_vector(h_cfg, state_cartesian, curvilinear))

    dummy = np.zeros(1)
    alg: TrainOfflineAlg = TrainOfflineAlg.create(jr.PRNGKey(0), dummy, dummy, nh, cfg)

    # Load ckpt.
    ckpt_dict = load_ckpt_ez(ckpt_path, {"alg": alg})
    alg = ckpt_dict["alg"]
    logger.info("Loaded ckpt from {}! update steps={}".format(ckpt_path, alg.update_idx))

    disc_gamma = alg.cfg.disc_gamma
    logger.info("   alg trained with disc_gamma: {}".format(disc_gamma))

    ii = 1
    T_x, T_obs, Th_h = dset.bT_x[ii], dset.bT_obs[ii], dset.bTh_h[ii]
    T_isterm = np.zeros(len(T_obs), dtype=bool)
    T_isterm[-1] = True

    T_obs_norm = (T_obs - alg.obs_mean) / alg.obs_std
    Th_Vh_resid = jax_jit_np(alg.value_net.apply)(T_obs_norm)
    Th_Vh_resid_ema = jax_jit_np(alg.get_ema)(T_obs_norm)


    # Th_Vh = Th_h + Th_Vh_resid
    # Th_Vh_ema = Th_h + Th_Vh_resid_ema
    Th_Vh = Th_Vh_resid
    Th_Vh_ema = Th_Vh_resid_ema

    max_gae_fn = ft.partial(get_max_gae_term, alg.cfg.disc_gamma, alg.cfg.gae_lambda)
    Th_Qh = jax_jit_np(max_gae_fn)(Th_h[:-1], Th_Vh, Th_h[:-1], T_isterm[1:])

    Th_h_disc = np.array(get_max_mc(disc_gamma, Th_h, Th_h))
    T_hdisc = Th_h_disc.max(axis=1)
    idx_last_safe = np.argmax(T_hdisc > 0) - 1

    fig, ax = plt.subplots()
    ax.axvline(idx_last_safe, lw=0.4, color="C4")
    ax.plot(Th_h[:, 0], lw=0.4, marker="o", color="0.2", ms=1, label="h")
    ax.plot(Th_h_disc[:, 0], lw=0.4, color="C0", alpha=0.8, label="h disc")
    ax.plot(Th_Vh[:, 0], lw=0.4, color="C1", alpha=0.8, label="Vh")
    ax.plot(Th_Vh_ema[:, 0], lw=0.4, color="C2", alpha=0.8, label="Vh ema")
    ax.plot(Th_Qh[:, 0], lw=0.4, color="C5", alpha=0.8, label="GAE tgt")
    line_labels(ax)
    fig_path = ckpt_path.parent.parent.parent / "compare_{}.pdf".format(ckpt_name)
    fig.savefig(fig_path, bbox_inches="tight")
    logger.info("Saved to {}!".format(fig_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
