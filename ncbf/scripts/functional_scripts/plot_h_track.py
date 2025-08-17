import functools as ft

import ipdb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from og.jax_utils import jax_jit_np, jax_vmap

from ncbf.ar_task import add_unsafe_eps
from ncbf.scripts.ncbf_config import get_cfgs


def main():
    obs_cfg, cfg = get_cfgs()

    b_ey = np.linspace(-2.1, 2.1, num=100000)

    def get_htrack(e_y):
        is_term = jnp.abs(e_y) >= cfg.track_width_term
        add_unsafe = ft.partial(add_unsafe_eps, margin_lo=cfg.margin_lo, margin_hi=cfg.margin_hi)

        h_track = (e_y) ** 2 - (cfg.track_width) ** 2
        h_track = add_unsafe(h_track)
        h_track = jnp.where(is_term, cfg.h_term + cfg.margin_hi, h_track)

        return h_track

    def get_htrack_original(e_y):
        h_track = e_y ** 2 - cfg.track_width ** 2
        return h_track

    b_htrack = jax_jit_np(jax_vmap(get_htrack))(b_ey)
    b_htrack_original = jax_jit_np(jax_vmap(get_htrack_original))(b_ey)

    fig, ax = plt.subplots(layout="constrained")
    # fig, ax = plt.subplots(layout="constrained", figsize=(2, 8))
    # ax.set_xticks(np.asarray([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]))
    ax.set_xlabel("ey")
    ax.set_ylabel("Avoidance Heuristic")

    b_htrack_filtered = b_htrack.copy()
    for i in range(1, len(b_htrack)):
        if abs(b_htrack[i] - b_htrack[i - 1]) > 0.05:
            b_htrack_filtered[i - 1] = np.nan  # Introduce NaN for the point before the jump
    ax.plot(b_ey, b_htrack_filtered, label='$h\'(x)$')
    # fig.savefig("h_track.png", bbox_inches="tight")

    ax.plot(b_ey, b_htrack_original, label='$h(x)$')
    # fig.savefig("h_track_original.png", bbox_inches="tight")

    # for closeup plot
    b_htrack_filtered_upper = b_htrack_filtered + 0.05
    b_htrack_filtered_lower = b_htrack_filtered - 0.05

    b_htrack_original_upper = b_htrack_original + 0.05
    b_htrack_original_lower = b_htrack_original - 0.05

    ax.fill_between(b_ey, b_htrack_filtered_lower, b_htrack_filtered_upper, color='blue', alpha=0.3, label='$h^\'(x) Error Margin$')
    ax.fill_between(b_ey, b_htrack_original_lower, b_htrack_original_upper, color='orange', alpha=0.3, label='$h(x) Error Margin$')

    ax.set_xlim(-1.6, -1.4)
    ax.set_ylim(-0.4, 0.3)

    ax.legend()
    fig.savefig("closeup_plot.png", bbox_inches="tight")

    # fig.savefig("h_track_comparison.png", bbox_inches="tight")



if __name__ == "__main__":
    main()
