from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from robot_planning.helper.convenience import get_drone_kinematics, get_drone_obstacles

try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser


class ObsInfo(NamedTuple):
    obs_pos: np.ndarray
    obs_radius: np.ndarray


def state_to_obs_drone(state: jnp.ndarray):
    # [ px pz theta vx vz omega ]
    assert state.shape == (6,)

    obs = get_drone_obstacles()
    n_obs = len(obs.obs_radius)
    assert n_obs >= 2, "Need at least two obstacles."

    px, pz, theta, vx, vz, omega = state
    pos2d = state[:2]

    # Compute the distance to the closest two obstacles.
    kin = get_drone_kinematics()

    # sdf
    o_dist = jnp.linalg.norm(pos2d - obs.obs_pos) - (obs.obs_radius + kin.get_radius())
    # Sort the distances.
    o_dist = jnp.sort(o_dist)
    # Take the two closest obstacles.
    o_dist_closest = o_dist[:2]

    # sincos encoding of theta.
    theta_sincos = jnp.array([jnp.sin(theta), jnp.cos(theta)])

    obs_state = jnp.array([px, pz, vx, vz, omega])
    obs = jnp.concatenate([obs_state, theta_sincos, o_dist_closest])

    assert obs.shape == (5 + 2 + 2,)

    return obs


def get_h_components(state):
    # [ px pz theta vx vz omega ]
    assert state.shape == (6,)

    obs = get_drone_obstacles()
    n_obs = len(obs.obs_radius)
    assert n_obs >= 2, "Need at least two obstacles."

    kin = get_drone_kinematics()

    px, pz, theta, vx, vz, omega = state
    pos2d = state[:2]

    # Obstacles.
    o_dist = jnp.linalg.norm(pos2d - obs.obs_pos, axis=-1) - (obs.obs_radius + kin.get_radius())
    assert o_dist.shape == (n_obs,)
    obs_dist_min = o_dist.min()

    # negative is safe.
    h_obs = -obs_dist_min

    # Clip the negative side so that we don't spend effort learning regions that are too safe.
    h_obs = jnp.clip(h_obs, -1.0, 0.0)

    # z >= 0, -z <= 0
    h_boundary = -pz  # should be approx [-1, 1] ?

    # is_oob = (theta < -np.pi / 2) | (np.pi / 2 < theta)
    h_drone_angle = jnp.abs(theta) - np.pi / 2
    h_drone_angle = h_drone_angle / (np.pi / 2)  # scale to [-1, 1]

    # Want px >= -4.5
    h_px_left = -(px + 4.5)

    # Just set all the unsafes to 1.
    def f(h_):
        eps = 0.3
        return jnp.where(h_ < 0, h_ - eps, 1.0)

    return {"obs": f(h_obs), "boundary": f(h_boundary), "drone_angle": f(h_drone_angle), "px_left": f(h_px_left)}


def get_h_vector_drone(state):
    h_components = get_h_components(state)
    h_list = list(h_components.values())
    return jnp.stack(h_list)
