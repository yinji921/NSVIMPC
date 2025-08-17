import functools as ft
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
from ncbf.avoid_utils import get_max_gae, get_max_mc
from ncbf.dset_offline import DsetOffline, S
from ncbf.offline.train_offline_alg import TrainOfflineAlg, TrainOfflineCfg
from ncbf.scripts.ncbf_config import get_cfgs
from robot_planning.helper.convenience import get_ccrf_track, plot_track
from robot_planning.helper.path_utils import get_root_dir


def main(ckpt_path: pathlib.Path):
    # Load dset.
    data_dir = get_root_dir() / "data"
    dset_path = data_dir / "dset.pkl"

    ckpt_name = ckpt_path.parent.name
    plot_dir = ckpt_path.parent.parent.parent

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

    print("cfg.Vh_act: {}".format(cfg.Vh_act))

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

    #######################################################################3
    bT_obs_norm = [(T_obs - alg.obs_mean) / alg.obs_std for T_obs in dset.bT_obs]

    ii = 1
    kk = 30

    T_x, T_obs_norm, Th_h = dset.bT_x[ii], bT_obs_norm[ii], dset.bTh_h[ii]

    Th_Vh_resid = jax_jit_np(alg.value_net.apply)(T_obs_norm)
    Th_Vh_resid_ema = jax_jit_np(alg.get_ema)(T_obs_norm)
    # Th_Vh = Th_h + Th_Vh_resid
    # Th_Vh_ema = Th_h + Th_Vh_resid_ema
    Th_Vh = Th_Vh_resid
    Th_Vh_ema = Th_Vh_resid_ema

    max_gae_fn = ft.partial(get_max_gae, disc_gamma, alg.cfg.gae_lambda)
    Tm1h_Qh = max_gae_fn(Th_h[:-1], Th_Vh, Th_h[:-1])
    Th_h_disc = np.array(get_max_mc(disc_gamma, Th_h, Th_h))
    T_hdisc = Th_h_disc.max(axis=1)
    idx_last_safe = np.argmax(T_hdisc > 0) - 1

    # Find all observations that are close to the obs.
    obs_norm = T_obs_norm[kk]

    b_length_cumsum = np.cumsum([0] + [len(T_obs) for T_obs in dset.bT_obs])[:-1]

    b_bb = np.concatenate([np.full(len(T_obs), ii) for ii, T_obs in enumerate(bT_obs_norm)], axis=0)
    b_obs_norm = np.concatenate(bT_obs_norm, axis=0)

    # Find the indices of the observations that are close to the obs.
    b_idx_sort = np.argsort(np.linalg.norm(b_obs_norm - obs_norm, axis=1))
    b_bb_sort = b_bb[b_idx_sort]
    b_kk_sort = b_idx_sort - b_length_cumsum[b_bb_sort]

    b_othertraj = b_idx_sort != ii
    b_bb_sort_other = b_bb_sort[b_othertraj]
    b_kk_sort_other = b_kk_sort[b_othertraj]

    # Get the distances to the obs from the same traj.
    T_dist_same = np.linalg.norm(T_obs_norm - obs_norm, axis=1)

    # Get the smallest distance to obs from other trajs.
    u_bb_unique, u_idxs = np.unique(b_bb_sort_other, return_index=True)
    u_bb, u_kk = b_bb_sort_other[u_idxs], b_kk_sort_other[u_idxs]
    u_obs = np.stack([bT_obs_norm[bb][kk] for bb, kk in zip(u_bb, u_kk)], axis=0)

    u_dist = np.linalg.norm(u_obs - obs_norm, axis=1)
    u_close_idx = np.arange(len(u_dist))[u_dist < np.quantile(u_dist, 0.2)]

    fig, ax = plt.subplots()
    ax.plot(T_dist_same, label="Same traj")
    ax.axhline(u_dist.min(), color="C1", label="Other traj min")
    ax.axhline(u_dist.mean(), color="C1", label="Other traj mean")
    for dist in u_dist[u_close_idx]:
        ax.axhline(dist, color="C3", alpha=0.3, lw=0.5)
    line_labels(ax)
    ax.set(xlabel="Steps", ylabel="Distance to normalized observation")
    fig_path = plot_dir / "obs_sim.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    ############################################################################
    # Plot the positions of the closest obs in the other trajs

    track = get_ccrf_track()

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.plot(T_x[:, S.X], T_x[:, S.Y], color="C0")
    ax.plot(T_x[0, S.X], T_x[0, S.Y], "s", ms=1.5, color="C0")
    ax.plot(T_x[kk, S.X], T_x[kk, S.Y], "o", ms=2.5, color="C0", zorder=10)

    # Plot the trajectories that are close to obs.
    for jj, uu in enumerate(u_close_idx):
        color = f"C{jj+1}"

        kk_other = u_kk[uu]
        ss = max(kk_other - 10, 0)

        T_x = dset.bT_x[u_bb[uu]]
        ax.plot(T_x[ss:, S.X], T_x[ss:, S.Y], color=color, alpha=0.3, lw=0.5)
        ax.plot(T_x[kk_other, S.X], T_x[kk_other, S.Y], "o", color=color, alpha=0.3, ms=1.2)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    plot_track(ax, track, 1.5, color="C0", lw=0.8, alpha=1.0, zorder=3)
    plot_track(ax, track, 2.0, color="C0", lw=0.8, alpha=1.0, zorder=3)
    ax.set(xlim=xlim, ylim=ylim)
    fig_path = plot_dir / "obs_sim_traj.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    ############################################################################
    # Finally, overlay the discounted h over the discounted h for Th_h.
    fig, ax = plt.subplots()
    ax.plot(Th_h[:, 0], lw=0.6, marker="o", color="0.2", ms=1, label="h")
    ax.plot(Th_h_disc[:, 0], lw=0.6, color="C0", alpha=0.8, label="h disc")
    ax.plot(Th_Vh[:, 0], lw=0.6, color="C1", alpha=0.8, label="Vh")
    ax.plot(Th_Vh_ema[:, 0], lw=0.6, color="C2", alpha=0.8, label="Vh ema")
    ax.plot(Tm1h_Qh[:, 0], lw=0.6, color="C4", alpha=0.8, label="Vh GAE")

    ax.plot(kk, Th_h_disc[kk, 0], marker="o", ms=1.8, color="C0")

    for jj, uu in enumerate(u_close_idx):
        color = f"C{jj+1}"

        kk_other = u_kk[uu]
        # Line up so that kk_other matches up with kk.
        ss = max(kk_other - kk, 0)
        bb = u_bb[uu]

        lh_h = dset.bTh_h[bb][ss:]
        lh_h_disc = get_max_mc(disc_gamma, lh_h, lh_h)
        ax.plot(lh_h_disc[:, 0], lw=0.4, color=color, alpha=0.3)

        ax.plot(kk, lh_h_disc[kk, 0], marker="o", ms=1.2, color=color)

    line_labels(ax)
    fig_path = ckpt_path.parent.parent.parent / "compare_{}.pdf".format(ckpt_name)
    fig.savefig(fig_path, bbox_inches="tight")
    logger.info("Saved to {}!".format(fig_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
