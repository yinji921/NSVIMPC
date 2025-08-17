import pathlib

import matplotlib.pyplot as plt
from loguru import logger
from og.path_utils import mkdir
from og.plot_utils import line_labels

from ncbf.offline.train_offline_alg import TrainOfflineAlg


def plot_comparison(
    idx: int, plot_dir: pathlib.Path, data_old: TrainOfflineAlg.EvalData, data_new: TrainOfflineAlg.EvalData
):
    fig, ax = plt.subplots(dpi=500)
    ax.plot(data_old.Th_h[:, 0], lw=0.4, marker="o", color="0.2", ms=1, label="h")
    ax.plot(data_old.Th_h_disc[:, 0], lw=0.4, color="C0", alpha=0.8, label="h disc")

    ax.plot(data_old.Th_Vh_eval[:, 0], lw=0.4, color="C1", alpha=0.8, label="OLD Vh")
    # ax.plot(data_old.Th_Qh_gae[:, 0], lw=0.4, color="C5", alpha=0.8, label="OLD GAE tgt")

    ax.plot(data_new.Th_Vh_eval[:, 0], lw=0.4, color="C2", alpha=0.8, label="NEW Vh")
    # ax.plot(data_new.Th_Qh_gae[:, 0], lw=0.4, color="C6", alpha=0.8, label="NEW GAE tgt")

    line_labels(ax)
    fig_path = plot_dir / f"compare/compare_{idx:08d}.jpg"
    mkdir(fig_path.parent)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved to {}!".format(fig_path))
