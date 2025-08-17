import functools as ft
import pathlib
import pickle

import ipdb
import typer
from loguru import logger

from ncbf.dset_offline_drone import DsetOfflineDrone


def main(pkl_paths: list[pathlib.Path], out: pathlib.Path):
    dsets = []
    for dset_path in pkl_paths:
        with open(dset_path, "rb") as f:
            # [ px pz theta vx vy w ]
            dset: DsetOfflineDrone = pickle.load(f)

            logger.info("Loaded {} with {} trajectories", dset_path, len(dset.bT_obs))
            dsets.append(dset)

    # Combine the lists.
    bT_obs = ft.reduce(lambda a, b: a + b, [dset.bT_obs for dset in dsets])
    bTh_h = ft.reduce(lambda a, b: a + b, [dset.bTh_h for dset in dsets])
    bT_x = ft.reduce(lambda a, b: a + b, [dset.bT_x for dset in dsets])
    bT_terminal = ft.reduce(lambda a, b: a + b, [dset.bT_terminal for dset in dsets])
    dset = DsetOfflineDrone(bT_obs, bTh_h, bT_x, bT_terminal)

    logger.info("Combined to {} trajectories", len(dset.bT_obs))

    # Save.
    with open(out, "wb") as f:
        pickle.dump(dset, f)

    logger.info("Saved to {}", out)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)
