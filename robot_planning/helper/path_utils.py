import pathlib


def get_root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent


def get_runs_dir() -> pathlib.Path:
    return get_root_dir() / "runs"


def get_commit_ckpt_dir() -> pathlib.Path:
    return get_root_dir() / "ncbf/ckpts"

def get_drone_commit_ckpt_dir() -> pathlib.Path:
    return get_root_dir() / "ncbf/ckpts_drone"

def get_experiments_dir() -> pathlib.Path:
    return get_root_dir() / "robot_planning/experiments"


def get_configs_dir() -> pathlib.Path:
    return get_root_dir() / "robot_planning/scripts/configs"
