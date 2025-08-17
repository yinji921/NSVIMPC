import ast

import matplotlib.pyplot as plt
import numpy as np

from robot_planning.environment.dynamics.autorally_dynamics.map_coords import MapCA
from robot_planning.environment.kinematics.simulated_kinematics import QuadrotorKinematics2D
from robot_planning.environment.kinematics.simulated_kinematics import PointKinematics
from robot_planning.factory.factories import kinematics_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.helper.utils import AUTORALLY_DYNAMICS_DIR, SCRIPTS_DIR

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser


def get_ccrf_track() -> MapCA:
    track_file_name = "ccrf_track_optimal.npz"
    track_path = AUTORALLY_DYNAMICS_DIR + "/" + track_file_name
    return MapCA(track_path)


def get_ccrf_track_with_obstacles() -> MapCA:
    track_file_name = "ccrf_track_optimal.npz"
    track_path = AUTORALLY_DYNAMICS_DIR + "/" + track_file_name
    obstacles_config_path = SCRIPTS_DIR + "/configs/reverse_run_Autorally_cluttered_env.cfg"
    obstacles_config_data = ConfigParser.ConfigParser()
    obstacles_config_data.read(obstacles_config_path)
    return MapCA(track_path, obstacles_config_data)


def get_drone_obstacles():
    obstacles_config_path = SCRIPTS_DIR + "/configs/run_quadrotor2d.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(obstacles_config_path)

    section_name = "my_collision_checker_for_collision"
    obstacles = np.asarray(ast.literal_eval(config_data.get(section_name, "obstacles")))
    obstacles_radius = np.asarray(ast.literal_eval(config_data.get(section_name, "obstacles_radius")))

    n_obs = len(obstacles_radius)
    assert obstacles.shape == (n_obs, 2)
    assert obstacles_radius.shape == (n_obs,)

    from ncbf.drone_task import ObsInfo

    return ObsInfo(obstacles, obstacles_radius)


def get_drone_goal_state():
    obstacles_config_path = SCRIPTS_DIR + "/configs/run_quadrotor2d.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(obstacles_config_path)

    section_name = "my_goal_checker1"
    goal_state = np.asarray(ast.literal_eval(config_data.get(section_name, "goal_state")))
    return goal_state


def get_drone_kinematics() -> QuadrotorKinematics2D:
    obstacles_config_path = SCRIPTS_DIR + "/configs/run_quadrotor2d.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(obstacles_config_path)

    section_name = "my_collision_checker_for_collision"
    kinematics_section_name = config_data.get(section_name, "kinematics")

    kinematics: QuadrotorKinematics2D = factory_from_config(
        kinematics_factory_base, config_data, kinematics_section_name
    )
    return kinematics

def get_dubins_kinematics() -> PointKinematics:
    obstacles_config_path = SCRIPTS_DIR + "/configs/run_dubins3d.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(obstacles_config_path)

    section_name = "my_collision_checker_for_collision"
    kinematics_section_name = config_data.get(section_name, "kinematics")

    kinematics: PointKinematics = factory_from_config(
        kinematics_factory_base, config_data, kinematics_section_name
    )
    return kinematics


def plot_track(ax: plt.Axes, track: MapCA, track_width: float, **kwargs):
    track_lines = []
    track_widths = [track_width, -track_width] if track_width > 0 else [0.0]
    for w_ in track_widths:
        xs = track.midpoints[0] + np.cos(track.heading + np.pi / 2) * w_
        ys = track.midpoints[1] + np.sin(track.heading + np.pi / 2) * w_
        # Make it a closed line by adding the first point to the end.
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        (line,) = ax.plot(xs, ys, **kwargs)
        track_lines.append(line)

    return track_lines


def plot_track_points(ax: plt.Axes, track: MapCA, **kwargs):
    xs = track.p[0]
    ys = track.p[1]
    # Make it a closed line by adding the first point to the end.
    xs = np.append(xs, xs[0])
    ys = np.append(ys, ys[0])
    (line,) = ax.plot(xs, ys, **kwargs)

    return line
