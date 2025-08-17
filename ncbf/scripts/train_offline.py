import pathlib
import pickle

import ipdb
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import typer
from attrs import define
from loguru import logger
from matplotlib.colors import CenteredNorm
from og.cfg_utils import Cfg
from og.ckpt_utils import get_ckpt_manager_sync
from og.cmap import get_BuRd
from og.jax_utils import jax2np, jax_jit_np, jax_vmap, merge01
from og.path_utils import mkdir
from og.plot_utils import line_labels
from og.schedules import Constant
from og.wandb_utils import reorder_wandb_name

import wandb
from ncbf.dset_offline import DsetOffline
from ncbf.offline.train_offline_alg import TrainOfflineAlg, TrainOfflineCfg, Traj
from robot_planning.helper.convenience import get_ccrf_track, plot_track
from robot_planning.helper.path_utils import get_root_dir


counter_clock_wise = 1
@define
class TrainerCfg(Cfg):
    n_iters: int
    log_every: int
    eval_every: int
    ckpt_every: int

    ckpt_max_keep: int = 100


track = None


def plot_eval(idx: int, plot_dir: pathlib.Path, data: TrainOfflineAlg.EvalData):
    global track
    if track is None:
        track = get_ccrf_track()

    nh = data.bbh_Vh.shape[2]

    figsize = np.array([5 * nh, 5.0])
    fig, axes = plt.subplots(1, nh, dpi=300, figsize=figsize)
    if nh == 1:
        axes = [axes]
    [ax.set_aspect("equal") for ax in axes]
    [plot_track(ax, track, 1.5, color="C0", lw=0.8, alpha=1.0, zorder=3) for ax in axes]
    [plot_track(ax, track, 2.0, color="C0", lw=0.8, alpha=1.0, zorder=3) for ax in axes]

    b_pos = merge01(data.bb_pos)
    bh_Vh = merge01(data.bbh_Vh)

    labels = ["Left", "Right"]
    # cmap = "RdBu_r"
    cmap = get_BuRd()
    for ii, ax in enumerate(axes):
        cm = ax.scatter(b_pos[:, 0], b_pos[:, 1], c=bh_Vh[:, ii], alpha=0.9, s=2, cmap=cmap, norm=CenteredNorm())
        fig.colorbar(cm, ax=ax)
        ax.set_title(labels[ii])
    fig_path = plot_dir / f"Vh/Vh_{idx:08d}.jpg"
    mkdir(fig_path.parent)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved to {}!".format(fig_path))

    # Compare learned Vh with discounted MC.
    idx_last_safe = np.argmax(data.Th_h_disc.max(axis=1) > 0) - 1

    fig, ax = plt.subplots()
    ax.axvline(idx_last_safe, lw=0.4, color="C4")
    ax.plot(data.Th_h[:, 0], lw=0.4, marker="o", color="0.2", ms=1, label="h")
    ax.plot(data.Th_h_disc[:, 0], lw=0.4, color="C0", alpha=0.8, label="h disc")
    ax.plot(data.Th_Vh_eval[:, 0], lw=0.4, color="C1", alpha=0.8, label="Vh")
    ax.plot(data.Th_Vh_eval_ema[:, 0], lw=0.4, color="C2", alpha=0.8, label="Vh ema")
    ax.plot(data.Th_Qh_gae[:, 0], lw=0.4, color="C5", alpha=0.8, label="GAE tgt")
    line_labels(ax)
    fig_path = plot_dir / f"evaltraj/evaltraj_{idx:08d}.jpg"
    mkdir(fig_path.parent)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved to {}!".format(fig_path))


def main(dset_path: pathlib.Path, wandb_name: str = None):
    # trainer_cfg = TrainerCfg(100_000, 100, 1_000, 1_000)
    trainer_cfg = TrainerCfg(300_000, 100, 1_000, 5_000)
    # T_sample = 96
    T_sample = 24
    # n_trajs = 96
    n_trajs = 256

    # Load dset.
    with open(dset_path, "rb") as f:
        dset: DsetOffline = pickle.load(f)

    # Extract a traj for evaluating.
    eval_ii = 1

    # Before we do anything, normalize the dset.
    dset, obs_mean, obs_std = dset.normalize()
    # remove trajectories that are too short
    dset = dset.remove_shorter_than(T_sample)

    T_obs_eval, Th_h_eval = dset.bT_obs[eval_ii], dset.bTh_h[eval_ii]
    dset = dset.pad_obs_final()

    # print(T_obs_eval)
    # exit()

    # hids = [256, 256]
    hids = [96, 96]
    # hids = [64, 64]
    # hids = [32, 32]
    lr = Constant(3e-4)
    wd = Constant(5e-2) # weight decay for nn, increasing it will alleviate overfitting of ncbf
    n_batches = 1
    # disc_gamma = 0.92 # discount factor, increasing it will augment unsafe zone
    # disc_gamma = 0.93 # discount factor, increasing it will augment unsafe zone
    disc_gamma = 0.94 # discount factor, increasing it will augment unsafe zone
    gae_lambda = 0.95
    # gae_lambda = 1.5
    ema_step = 1e-3

    _, nh = dset.bTh_h[0].shape

    # Vh_act = "softplus"
    Vh_act = "identity"
    cfg = TrainOfflineCfg("relu", Vh_act, hids, lr, wd, n_batches, disc_gamma, gae_lambda, ema_step)
    alg = TrainOfflineAlg.create(jr.PRNGKey(123456), obs_mean, obs_std, nh, cfg)

    run = wandb.init(project="ar_cbf_offline", config=cfg.asdict())
    reorder_wandb_name(wandb_name)

    run_dir = get_root_dir() / "runs/offline" / run.name

    run_dir.mkdir(exist_ok=True, parents=True)

    plot_dir = mkdir(run_dir / "plot")
    ckpt_dir = mkdir(run_dir / "ckpts")

    ckpt_manager = get_ckpt_manager_sync(ckpt_dir.absolute(), max_to_keep=trainer_cfg.ckpt_max_keep)

    rng = np.random.default_rng(seed=12345)
    for idx in range(trainer_cfg.n_iters):
        should_log = idx % trainer_cfg.log_every == 0
        should_eval = idx % trainer_cfg.eval_every == 0
        should_ckpt = idx % trainer_cfg.ckpt_every == 0

        # Sample random subset of trajs.
        bT_traj = dset.sample_trajs(n_trajs, T_sample, rng, replace=False, p_final=0.1)
        # Update.
        alg, loss_info = alg.update(bT_traj)

        if should_log:
            log_dict = {f"train/{k}": v for k, v in loss_info.items()}
            logger.info("{:5} | Loss={:.2e}".format(idx, loss_info["loss"]))
            wandb.log(log_dict, step=idx)

        if should_eval:
            logger.info("Eval...")
            data = jax2np(alg.eval(T_obs_eval, Th_h_eval, counter_clock_wise = counter_clock_wise))
            logger.info("Eval... Done!")
            plot_eval(idx, plot_dir, data)

            log_dict = {f"eval/{k}": v for k, v in data.info.items()}
            wandb.log(log_dict, step=idx)

        if should_ckpt:
            ckpt_manager.save_ez(idx, {"alg": alg, "cfg": cfg.asdict()})
            logger.info("Saved ckpt!")

    # Save at the end.
    ckpt_manager.save_ez(idx, {"alg": alg, "cfg": cfg.asdict()})


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
