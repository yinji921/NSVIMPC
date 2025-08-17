import jax
import functools as ft
import jax.numpy as jnp
import numpy as np
from attrs import define
from og.angle_utils import wrap_to_pi
from og.cfg_utils import Cfg
import ipdb
import ast
from robot_planning.helper.path_utils import get_configs_dir
from robot_planning.environment.dynamics.autorally_dynamics.map_coords import MapCA
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser

@define
class ObsCfg(Cfg):
    # How densely to sample the track.
    kappa_dt: float
    # How many kappas to sample.
    n_kappa: int
    # Exponent. Values larger than 1 will have the s near the end be sampled more loosely.
    dt_exp: float


@define
class ConstrCfg(Cfg):
    track_width: float
    # When the track width exceeds this value, the simulation is considered unsafe and the run is terminated.
    track_width_term: float
    margin_lo: float
    margin_hi: float
    h_term: float


def state_to_obs(track: MapCA, cfg: ObsCfg, state, curvilinear):
    # [ vx vy wz wF wR Psi x y]
    assert state.shape == (8,)
    # [ e_psi e_y s ]
    assert curvilinear.shape == (3,)

    vx, vy, wz, wF, wR, psi, x, y = state
    e_psi, e_y, s = curvilinear

    # Compute list of s. Approximate with constant velocity.
    b_dt = (jnp.arange(cfg.n_kappa) ** cfg.dt_exp) * cfg.kappa_dt
    b_s = s + jnp.cos(e_psi) * vx * b_dt

    #   Wrap to [0, s[-1]], otherwise we're querying outside the track.
    b_s = jnp.mod(b_s, track.s[-1])

    # Get track kappa at each s.
    if False:
        # b_x0, b_y0, b_theta0: (n_kappa, 1)
        # b_kappa: (n_kappa, 1, 1)
        b_x0, b_y0, b_theta0, _, b_kappa = jax.vmap(track.get_cur_reg_from_s)(b_s)
        b_x0, b_y0, b_theta0 = b_x0.squeeze(1), b_y0.squeeze(1), b_theta0.squeeze(1)
        b_kappa = b_kappa.squeeze((1, 2))
    else:
        b_x0, b_y0, b_theta0, _, b_kappa = jax.vmap(track.get_lerp_from_s)(b_s)
    b_pos = jnp.stack([b_x0, b_y0], axis=1)

    # Augment with the angle to the next waypoint.
    b_angle2next = wrap_to_pi(b_theta0[1:] - b_theta0[:-1])
    # Augment with the distance to the next waypoint.
    b_dist2next = jnp.linalg.norm(b_pos[1:] - b_pos[:-1], axis=1)

    # sincos encode e_psi.
    e_psi_sin, e_psi_cos = jnp.sin(e_psi), jnp.cos(e_psi)

    state_dyn = jnp.array([vx, vy, wz, wF, wR, e_psi_sin, e_psi_cos, e_y])
    # state_dyn = jnp.array([vx, vy, wz, e_psi_sin, e_psi_cos, e_y])
    obs = jnp.concatenate([state_dyn, b_kappa, b_angle2next, b_dist2next], axis=0)
    # assert obs.shape == (6 + cfg.n_kappa + 2 * (cfg.n_kappa - 1),)
    assert obs.shape == (8 + cfg.n_kappa + 2 * (cfg.n_kappa - 1),)
    return obs


def add_unsafe_eps(h, margin_lo: float, margin_hi: float):
    return jnp.where(h < 0, h - margin_lo, h + margin_hi)

def get_h_with_obstacles(state_cartesian, h):
    clock_wise = False  # Always false
    if clock_wise:
        config_path = get_configs_dir() / "run_Autorally_CBF_MPPI_for_experiment.cfg"
    else:  # counter-clockwise
        config_path = get_configs_dir() / "reverse_run_Autorally_CBF_MPPI_for_experiment.cfg"
        # config_path = get_configs_dir() / "reverse_run_Autorally_cluttered_env.cfg"

    # for 2d drone
    config_path = get_configs_dir() / "run_quadrotor2d.cfg"
    config_data = ConfigParser.ConfigParser()
    config_data.read(config_path)
    obstacles = np.asarray(ast.literal_eval(config_data.get("my_collision_checker_for_collision", "obstacles")))
    obstacles_radius = np.asarray(
        ast.literal_eval(config_data.get("my_collision_checker_for_collision", "obstacles_radius")))

    state_cartesian = state_cartesian[6:8]
    distance_to_obstacles = jnp.linalg.norm(obstacles - state_cartesian, axis=1)
    obstacles_danger = obstacles_radius ** 2 - distance_to_obstacles ** 2
    obstacles_danger = jnp.max(obstacles_danger)
    h_with_obstacles = jnp.where(obstacles_danger > h, obstacles_danger, h)
    return h_with_obstacles


def get_h_components(cfg: ConstrCfg, state_cartesian, curvilinear):
    # [ vx vy wz wF wR ]
    assert state_cartesian.shape == (8,)
    # [ e_psi e_y s ]
    assert curvilinear.shape == (3,)
    add_unsafe = ft.partial(add_unsafe_eps, margin_lo=cfg.margin_lo, margin_hi=cfg.margin_hi)

    e_psi, e_y, s = curvilinear

    is_term = jnp.abs(e_y) >= cfg.track_width_term # True if crash

    h_track = e_y ** 2 - cfg.track_width ** 2 # h_track > 0 if collision

    h_track = get_h_with_obstacles(state_cartesian, h_track)

    h_track = add_unsafe(h_track)
    h_track = jnp.where(is_term, cfg.h_term + cfg.margin_hi, h_track)

    return {"track": h_track}


def get_h_vector(cfg, state_cartesian, curvilinear):
    h_components = get_h_components(cfg, state_cartesian, curvilinear)
    h_list = list(h_components.values())
    return jnp.stack(h_list)
