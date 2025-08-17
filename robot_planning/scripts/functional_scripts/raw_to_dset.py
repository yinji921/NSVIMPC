import functools as ft
import pathlib
import pickle

import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import typer
from loguru import logger
from og.jax_utils import jax_jit_np, jax_vmap

from ncbf.ar_task import ConstrCfg, ObsCfg, get_h_vector, state_to_obs
from ncbf.avoid_utils import get_max_mc
from ncbf.dset_offline import DsetOffline, S
from ncbf.scripts.ncbf_config import get_cfgs
from robot_planning.helper.convenience import get_ccrf_track, plot_track, get_ccrf_track_with_obstacles


def main(pkl_path: pathlib.Path):
    with open(pkl_path, "rb") as f:
        # [ vx vy wz wF wR psi X Y e_psi e_y s ]
        bxT_x = pickle.load(f)

    track = get_ccrf_track()

    # track = get_ccrf_track_with_obstacles()

    plot_dir = pkl_path.parent

    # Convert each trajectory to observations.
    # n_kappa = 10
    # obs_cfg = ObsCfg(kappa_dt=0.2, n_kappa=n_kappa, dt_exp=1.25)
    # h_cfg = ConstrCfg(track_width=1.3, track_width_term=1.9, margin_lo=0.2, margin_hi=1.0, h_term=2.2)
    obs_cfg, h_cfg = get_cfgs()
    n_kappa = obs_cfg.n_kappa

    ii = 1
    T_x = bxT_x[ii].T
    # state_to_obs(track, obs_cfg, T_x[0, :5], T_x[0, -3:])
    # input()
    T_obs = jax_vmap(ft.partial(state_to_obs, track, obs_cfg))(T_x[:, :8], T_x[:, -3:])
    Th_h = jax_vmap(ft.partial(get_h_vector, h_cfg))(T_x[:, :8], T_x[:, -3:])

    disc_gamma = 0.92
    Th_h_disc = np.array(get_max_mc(disc_gamma, Th_h, Th_h))
    T_hdisc = Th_h_disc.max(axis=1)
    idx_last_safe = np.argmax(T_hdisc > 0) - 1

    kk = 0
    # kk = 15
    # kk = 80
    b_dt = (jnp.arange(obs_cfg.n_kappa) ** obs_cfg.dt_exp) * obs_cfg.kappa_dt
    print("dt: {}".format(b_dt))
    b_s = T_x[kk, S.S] + jnp.cos(T_x[kk, S.EPSI]) * T_x[kk, S.VX] * b_dt
    b_s = jnp.mod(b_s, track.s_total)

    b_x0, b_y0, b_theta0, _, b_kappa = jax_vmap(track.get_lerp_from_s)(b_s)

    # ax.plot(T_x[:, S.S])
    # ax.axhline(track.s[-1])

    logger.info("T_x.shape: {}".format(T_x.shape))

    figsize = np.array([14.0, 14.0])
    fig, axes = plt.subplots(5, height_ratios=[1, 1, 1, 1, 4], figsize=figsize)

    axes[0].plot(T_obs[kk, 8 : 8 + n_kappa], lw=0.5, marker="o")
    axes[0].set_ylabel(r"$\kappa$")

    axes[1].plot(T_obs[kk, 8 + n_kappa : 8 + 2 * n_kappa - 1], lw=0.5, marker="o")
    axes[1].set_ylabel(r"$\Delta \theta$")

    axes[2].plot(T_x[:, S.EY])
    axes[2].set_ylabel(r"$e_y$")

    axes[3].axhline(h_cfg.h_term + h_cfg.margin_hi, color="C5", lw=0.6, alpha=0.9)
    axes[3].axvline(idx_last_safe, lw=0.4, color="C4")
    axes[3].plot(Th_h[:, 0], lw=0.4, marker="o", ms=1, color="C0", label="Left")
    # axes[3].plot(Th_h[:, 1], lw=0.4, marker="o", ms=1, color="C1", label="Right")
    axes[3].plot(Th_h_disc[:, 0], lw=0.4, color="C0", alpha=0.8)
    # axes[3].plot(Th_h_disc[:, 1], lw=0.4, color="C1", alpha=0.8)
    axes[3].axhline(0.0, color="0.3", lw=0.4, alpha=0.7)
    # axes[3].legend(ncol=2, bbox_to_anchor=(0.5, 1.0), loc="lower center")
    axes[3].set_ylabel(r"$h$")

    plot_track(axes[-1], track, 0.0, color="0.1", lw=0.8, alpha=0.3, zorder=2.9)
    plot_track(axes[-1], track, 1.5, color="C0", lw=0.8, alpha=1.0, zorder=3)
    plot_track(axes[-1], track, 2.0, color="C0", lw=0.8, alpha=1.0, zorder=3)
    axes[-1].plot(
        T_x[:, S.X], T_x[:, S.Y], lw=0.4, marker="o", ms=1, mfc="0.1", mec="none", alpha=0.3, color="C1", zorder=4
    )
    axes[-1].plot(T_x[-1, S.X], T_x[-1, S.Y], marker="o", ms=2, color="C1", zorder=5)
    axes[-1].plot(T_x[idx_last_safe, S.X], T_x[idx_last_safe, S.Y], marker="s", ms=2, color="C4", zorder=5)
    axes[-1].set(aspect="equal")

    axes[-1].plot(T_x[kk, S.X], T_x[kk, S.Y], marker="s", ms=2.5, color="C3", alpha=0.7, zorder=4.5)
    axes[-1].scatter(b_x0, b_y0, color="C5", s=2, zorder=6)

    fig.savefig(plot_dir / "obs.pdf", bbox_inches="tight")
    plt.close(fig)
    # ------------------------------------------------------

    fig, ax = plt.subplots()
    plot_track(ax, track, 0.0, color="0.1", lw=0.8, alpha=1.0, zorder=3)
    plot_track(ax, track, 1.5, color="C0", lw=0.8, alpha=1.0, zorder=3)
    plot_track(ax, track, 2.0, color="C0", lw=0.8, alpha=1.0, zorder=3)

    for xT_x in bxT_x:
        T_x = xT_x.T
        ax.plot(T_x[:, S.X], T_x[:, S.Y], ls="none", marker="o", ms=1.5, mfc="C1", mec="none", alpha=0.2)
    fig.savefig(plot_dir / "density.pdf", bbox_inches="tight")
    plt.close(fig)

    ###############################################################
    bT_obs, bTh_h, bT_x = [], [], []
    for xT_x in tqdm.tqdm(bxT_x):
        T_x = xT_x.T
        T_obs = jax_jit_np(jax_vmap(ft.partial(state_to_obs, track, obs_cfg)))(T_x[:, :8], T_x[:, -3:])
        Th_h = jax_jit_np(jax_vmap(ft.partial(get_h_vector, h_cfg)))(T_x[:, :8], T_x[:, -3:])

        bT_obs.append(T_obs)
        bTh_h.append(Th_h)
        bT_x.append(T_x)

    n_samples = sum([len(T_obs) for T_obs in bT_obs])
    T_min = min([len(T_obs) for T_obs in bT_obs])
    T_max = max([len(T_obs) for T_obs in bT_obs])
    # This is the dset used for training
    dset = DsetOffline(bT_obs, bTh_h, bT_x)
    dset_path = plot_dir / "dset.pkl"
    with open(dset_path, "wb") as f:
        pickle.dump(dset, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved dset to {}! #Samples = {}, T min={}, max={}".format(dset_path, n_samples, T_min, T_max))

    # ###############################################################
    # Plot distribution of the states.
    b_obs = np.concatenate(bT_obs, axis=0)

    state_idxs = [0, 1, 2, 3, 4, 7]
    labels = [r"$v_x$", r"$v_y$", r"$\omega$", r"$\omega_F$", r"$\omega_R$", r"$e_y$"]

    fig, axes = plt.subplots(len(state_idxs))
    for idx, label, ax in zip(state_idxs, labels, axes):
        ax.hist(b_obs[:, idx], bins=64, color="C1")
        ax.set_ylabel(label, rotation=0, ha="right")
    fig.savefig(plot_dir / "state_dist.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
