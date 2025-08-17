import pathlib
from loguru import logger
import pickle

import ipdb
import matplotlib.pyplot as plt
import numpy as np
import typer

from ncbf.dset_offline import DsetOffline


def main(dset_path: pathlib.Path):
    with open(dset_path, "rb") as f:
        dset: DsetOffline = pickle.load(f)

    b_obs = np.concatenate(dset.bT_obs, axis=0)

    b_vx = b_obs[:, 0]
    b_vy = b_obs[:, 1]
    b_v = np.sqrt(b_vx**2 + b_vy**2)
    b_wF = b_obs[:, 3]
    b_wR = b_obs[:, 4]

    figsize = np.array([8.0, 6.0])
    fig, axes = plt.subplots(2, 3, figsize=figsize, layout="constrained")
    axes[0, 0].scatter(b_vx, b_wF, s=1)
    axes[0, 0].set(xlabel="vx", ylabel="wF")
    axes[0, 1].scatter(b_vy, b_wF, s=1)
    axes[0, 1].set(xlabel="vy", ylabel="wF")
    axes[0, 2].scatter(b_v, b_wF, s=1)
    axes[0, 2].set(xlabel="v", ylabel="wF")

    axes[1, 0].scatter(b_vx, b_wR, s=1)
    axes[1, 0].set(xlabel="vx", ylabel="wR")
    axes[1, 1].scatter(b_vy, b_wR, s=1)
    axes[1, 1].set(xlabel="vy", ylabel="wR")
    axes[1, 2].scatter(b_v, b_wR, s=1)
    axes[1, 2].set(xlabel="v", ylabel="wR")

    plot_dir = dset_path.parent
    fig_path = plot_dir / "wf_wr_correlation.pdf"
    fig.savefig(fig_path, bbox_inches="tight")

    logger.info("Saved to {}".format(fig_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
