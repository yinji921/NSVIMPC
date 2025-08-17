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

from ncbf.avoid_utils import get_max_mc
from ncbf.drone_task import get_h_vector_drone, state_to_obs_drone
from ncbf.dset_offline import DsetOffline, S
from ncbf.dset_offline_drone import DsetOfflineDrone
from ncbf.scripts.ncbf_config import get_cfgs
from robot_planning.helper.convenience import (get_ccrf_track, get_ccrf_track_with_obstacles, get_drone_goal_state,
                                        get_drone_obstacles, plot_track)


def main(dset_path: pathlib.Path):
    """For drone."""
    plot_dir = dset_path.parent
    with open(dset_path, "rb") as f:
        # [ px pz theta vx vy w ]
        dset: DsetOfflineDrone = pickle.load(f)

    obs_info = get_drone_obstacles()

    # ###############################################################
    # Plot distribution of the states.
    bT_x = dset.bT_x
    b_x = np.concatenate(bT_x, axis=0)

    state_idxs = [0, 1, 2, 3, 4, 5]
    labels = [r"$p_x$", r"$p_z$", r"$\theta$", r"$v_x$", r"$v_z$", r"$\omega$"]

    fig, axes = plt.subplots(len(state_idxs), layout="constrained")
    for idx, label, ax in zip(state_idxs, labels, axes):
        ax.hist(b_x[:, idx], bins=64, color="C1")
        ax.set_ylabel(label, rotation=0, ha="right")
    fig.savefig(plot_dir / f"state_dist_drone.pdf", bbox_inches="tight")
    plt.close(fig)

    # Also do a 2d histogram of the position.
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    h = ax.hist2d(b_x[:, 0], b_x[:, 1], bins=64, cmap="viridis", cmin=1)

    # Plot the obstacles.
    for ii, pos in enumerate(obs_info.obs_pos):
        ax.add_patch(plt.Circle(pos, obs_info.obs_radius[ii], color="C3", alpha=0.5))

    fig.colorbar(h[3], ax=ax)
    fig.savefig(plot_dir / "state_dist_drone_pos.pdf", bbox_inches="tight")
    plt.close(fig)

    # --------------------------------
    # How is it possible that the x position is reaching such high values?
    # Find a trajectory that has px >= 8.
    # for ii, T_x in enumerate(bT_x):
    #     if np.any(T_x[:, 3] >= 8):
    #         break
    h_labels = ["obs", "boundary", "drone_angle", "px_left"]

    # find an index where the final px >= 5.
    for ii, T_x in enumerate(bT_x):
        if T_x[-1, 0] >= 5:
            break

    print("ii: {}".format(ii))

    T_x = bT_x[ii]
    Th_h = dset.bTh_h[ii]

    # Plot the trajectory.
    nx = len(state_idxs)
    nh = Th_h.shape[1]

    figsize = np.array([10.0, 2.0 * (nx + nh)])

    fig, axes = plt.subplots(nx + nh, layout="constrained", figsize=figsize)
    for ii, ax in enumerate(axes[:nx]):
        ax.plot(T_x[:, ii])
        ax.set_ylabel(labels[ii], rotation=0, ha="right")
    for ii, ax in enumerate(axes[nx:]):
        ax.plot(Th_h[:, ii])
        ax.set_ylabel(h_labels[ii], rotation=0, ha="right")
    fig.savefig(plot_dir / f"drone_traj.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------
    # Plot the trajectory in position space.
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(T_x[:, 0], T_x[:, 1])
    # Plot obstacles.
    for ii, pos in enumerate(obs_info.obs_pos):
        ax.add_patch(plt.Circle(pos, obs_info.obs_radius[ii], color="C3", alpha=0.5))
    ax.set_aspect("equal")
    fig.savefig(plot_dir / "drone_traj2d.pdf", bbox_inches="tight")

    ipdb.set_trace()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
