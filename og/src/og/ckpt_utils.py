import datetime
import pathlib
from typing import Any

import attrs
import jax
import numpy as np
import orbax
import orbax.checkpoint
from attrs import asdict
from flax.training import orbax_utils
from orbax.checkpoint import CheckpointManager


def save_ckpt_ez(save_path: pathlib.Path, item: Any):
    ckpter = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(item)
    ckpter.save(save_path, item, save_args=save_args)


def load_ckpt_ez(load_path: pathlib.Path, item: Any):
    ckpter = orbax.checkpoint.PyTreeCheckpointer()
    return ckpter.restore(load_path, item=item)


class WrappedCkptManager(CheckpointManager):
    def save_ez(self, step: int, items: Any):
        if isinstance(items, dict):
            # Replacce all attrs dataclasses with their dict equivalents.
            items_ = {}
            for k, v in items.items():
                if attrs.has(v):
                    v = asdict(v)
                items_[k] = v
            items = items_

        save_args = orbax_utils.save_args_from_target(items)
        return self.save(step, items, save_kwargs={"save_args": save_args})


def get_ckpt_manager(ckpt_dir: pathlib.Path, max_to_keep: int = 100):
    # Get random port.
    random_port = np.random.randint(10000, 20000)
    jax.distributed.initialize("localhost:{}".format(random_port), num_processes=1, process_id=0)
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=max_to_keep, keep_time_interval=datetime.timedelta(minutes=5), create=True
    )
    checkpointer = orbax.checkpoint.PyTreeCheckpointHandler()
    async_checkpointer = orbax.checkpoint.AsyncCheckpointer(checkpointer, timeout_secs=50)
    ckpt_manager = orbax.checkpoint.CheckpointManager(ckpt_dir, async_checkpointer, options)
    return ckpt_manager


def get_checkpointer():
    return orbax.checkpoint.PyTreeCheckpointer()


def get_ckpt_manager_sync(ckpt_dir: pathlib.Path, max_to_keep: int = 50, minutes: float = 5):
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        keep_time_interval=datetime.timedelta(minutes=minutes),
        create=True,
        step_format_fixed_length=8,
    )
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt_manager = WrappedCkptManager(ckpt_dir, orbax_checkpointer, options)
    return ckpt_manager


def get_create_args_path(ckpt_dir: pathlib.Path):
    return ckpt_dir / "create_args.pkl"


def get_ckpt_dir_from_path(ckpt_path: pathlib.Path):
    # Either ckpt_path points to ckpts, or it points to a subdirectory in ckpts.
    if ckpt_path.name == "ckpts":
        return ckpt_path

    assert ckpt_path.parent.name == "ckpts"
    return ckpt_path.parent
