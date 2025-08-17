import copy
from functools import partial

import ipdb
import jax.numpy as np
import jax
import ast

from robot_planning.environment.collision_checker import CollisionChecker
from robot_planning.environment.dynamics.autorally_dynamics.map_coords import MapCA
from robot_planning.factory.factories import (
    collision_checker_factory_base,
    goal_checker_factory_base,
)
from robot_planning.factory.factory_from_config import factory_from_config
# from robot_planning.environment.barriers.barrier_net import BarrierNN
from robot_planning.factory.factories import barrier_net_factory_base
import functools as ft
import jax.numpy as jnp
# from robot_planning.environment.ncbf import NCBF
from robot_planning.environment.dynamics.autorally_dynamics.autorally_dynamics import AutoRallyDynamics
import loguru
from ncbf.ar_task import ConstrCfg, ObsCfg, get_h_vector, state_to_obs
from ncbf.scripts.ncbf_config import get_cfgs, get_h_cfg_for


class CostEvaluator:
    def __init__(self, goal_checker=None, collision_checker=None):
        self.goal_checker = goal_checker
        self.collision_checker: CollisionChecker = collision_checker

    def initialize_from_config(self, config_data, section_name):
        raise NotImplementedError

    def evaluate(self, state_cur, state_next=None, dyna_obstacle_list=None):
        raise NotImplementedError

    def set_collision_checker(self, collision_checker=None):
        self.collision_checker = collision_checker

    def set_goal_checker(self, goal_checker=None):
        self.goal_checker = goal_checker

class QuadraticCostEvaluator(CostEvaluator):
    def __init__(
        self,
        goal_checker=None,
        collision_checker=None,
        Q=None,
        R=None,
        collision_cost=None,
        goal_cost=None,
    ):
        CostEvaluator.__init__(self, goal_checker, collision_checker)
        self.Q = Q
        self.R = R
        self.collision_cost = collision_cost
        self.goal_cost = goal_cost
        self.check_other_agents = False

    def initialize_from_config(self, config_data, section_name):
        if config_data.has_option(section_name, "collision_cost"):
            self.collision_cost = config_data.getfloat(
                section_name, "collision_cost"
            )  # collision_cost should be positive
        if config_data.has_option(section_name, "goal_cost"):
            self.goal_cost = config_data.getfloat(
                section_name, "goal_cost"
            )  # goal_cost should be negative
        if config_data.has_option(section_name, "Q"):
            Q = np.asarray(ast.literal_eval(config_data.get(section_name, "Q")))
            if Q.ndim == 1:
                self.Q = np.diag(Q)
            else:
                self.Q = Q
        if config_data.has_option(section_name, "QN"):
            QN = np.asarray(ast.literal_eval(config_data.get(section_name, "QN")))
            if QN.ndim == 1:
                self.QN = np.diag(QN)
            else:
                self.QN = QN
        else:
            self.QN = self.Q
        if config_data.has_option(section_name, "R"):
            R = np.asarray(ast.literal_eval(config_data.get(section_name, "R")))
            if R.ndim == 1:
                self.R = np.diag(R)
            else:
                self.R = R
        if config_data.has_option(section_name, "goal_checker"):
            goal_checker_section_name = config_data.get(section_name, "goal_checker")
            self.goal_checker = factory_from_config(
                goal_checker_factory_base, config_data, goal_checker_section_name
            )
        if config_data.has_option(section_name, "collision_checker"):
            collision_checker_section_name = config_data.get(
                section_name, "collision_checker"
            )
            self.collision_checker = factory_from_config(
                collision_checker_factory_base,
                config_data,
                collision_checker_section_name,
            )

        if config_data.has_option(section_name, "check_other_agents"):
            self.check_other_agents = config_data.getboolean(
                section_name, "check_other_agents"
            )

    def evaluate(self, state_cur, actions=None, dyna_obstacle_list=None, dynamics=None, opponent_agents=None):
        if state_cur.ndim == 1:
            state_cur = state_cur.reshape((-1, 1))


        error_state_right = np.expand_dims(
            (state_cur - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=2
        )
        error_state_left = np.expand_dims(
            (state_cur - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=1
        )
        # 1/2 xQx
        cost = (
                (1 / 2)
                * error_state_left
                @ np.tile(np.expand_dims(self.Q, axis=0), (state_cur.shape[1], 1, 1))
                @ error_state_right
        )
        if actions is not None:
            actions_left = np.expand_dims(actions.T, axis=1)
            actions_right = np.expand_dims(actions.T, axis=2)
        # 1/2 uRu
            cost += (
                    (1 / 2)
                    * actions_left
                    @ np.tile(np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1))
                    @ actions_right
            )

        collisions = self.collision_checker.check(
            state_cur, opponent_agents=opponent_agents
        )
        collisions = collisions.reshape((-1, 1, 1))
        if self.collision_cost is not None:
            cost += collisions * self.collision_cost
        else:
            cost += collisions * 1000  # default collision cost

        return cost

    def evaluate_terminal_cost(
        self, state_cur, actions=None, dyna_obstacle_list=None, dynamics=None
    ):
        return self.evaluate(state_cur, actions, dyna_obstacle_list, dynamics)


class Quadrotor2DCBFCostEvaluator(QuadraticCostEvaluator):
    def __init__(
        self,
        cbf_alpha=0.9,
        goal_checker=None,
        collision_checker=None,
        Q=None,
        R=None,
        collision_cost=None,
        goal_cost=None,
    ):
        self.cbf_alpha = cbf_alpha
        QuadraticCostEvaluator.__init__(
            self, goal_checker, collision_checker, Q, R, collision_cost, goal_cost
        )
        self.cbf_alpha_resample = 0.5

    def initialize_from_config(self, config_data, section_name):
        # Get superclass parameters
        QuadraticCostEvaluator.initialize_from_config(
            self, config_data, section_name
        )
        # Get CBF-specific parameters
        self.cbf_alpha = config_data.getfloat(section_name, "cbf_alpha")
        barrier_net_section_name = config_data.get(section_name, "barrier_net")
        # self.barrier_nn = factory_from_config(
        #     barrier_net_factory_base, config_data, barrier_net_section_name
        # )
        self.include_cbf_cost = config_data.getboolean(section_name, "include_cbf_cost")
        self.cbf_vio_cost = 200.0
        self.cbf_safe_cost = 10.0
        if config_data.has_option(section_name,"include_cbf_cost"):
            self.include_cbf_cost = config_data.getboolean(section_name, "include_cbf_cost")
        else:
            self.include_cbf_cost = True

        self.obstacles = np.asarray(ast.literal_eval(config_data.get("my_collision_checker_for_collision", "obstacles")))
        self.obstacles_radius = np.asarray(
            ast.literal_eval(config_data.get("my_collision_checker_for_collision", "obstacles_radius")))

    def get_h_(self, state: jnp.ndarray) -> jnp.ndarray:
        # Get the base h value.
        h_h = self.get_h_vector_(state)
        return h_h

    def get_h_vector_(self, state: jnp.ndarray):
        obs_cfg, h_cfg = get_cfgs()
        assert state.shape == (6,)
        h_h = self.get_h_vector(h_cfg, state)
        return h_h

    def get_h_vector(self, cfg, state_cartesian):
        h_components = self.get_h_components(cfg, state_cartesian)
        h_list = list(h_components.values())
        return jnp.stack(h_list)

    def add_unsafe_eps(self, h, margin_lo: float, margin_hi: float):
        return jnp.where(h < 0, h - margin_lo, h + margin_hi)

    def get_h_components(self, cfg: ConstrCfg, state_cartesian):
        # [ vx vy wz wF wR ]
        assert state_cartesian.shape == (6,)
        # [ e_psi e_y s ]
        add_unsafe = ft.partial(self.add_unsafe_eps, margin_lo=cfg.margin_lo, margin_hi=cfg.margin_hi)

        h_track = -10*state_cartesian[1] # making height > 0 to be safe TODO: the coefficient needs to be tuned

        h_track = self.get_h_with_obstacles(state_cartesian, h_track)

        h_track = add_unsafe(h_track)

        return {"track": h_track}

    def get_h_with_obstacles(self, state_cartesian, h):
        state_cartesian = state_cartesian[:2]
        distance_to_obstacles = jnp.linalg.norm(self.obstacles - state_cartesian, axis=1)
        obstacles_danger = self.obstacles_radius ** 2 - distance_to_obstacles ** 2
        obstacles_danger = jnp.max(obstacles_danger)
        h_with_obstacles = jnp.where(obstacles_danger > h, obstacles_danger, h)
        return h_with_obstacles

    def evaluate_cost(self, state_cur, state_next, action, noise, dynamics: AutoRallyDynamics):
        """
        :param state_cur: (nx, ) = (8, )
        :param state_next: (nx, ) = (8, )
        :param action: (nu, ) = (2, )
        :param noise: (nu, ) = (2, )
        :param dynamics:
        :param ncbf_weights: Weights for the NCBF to use instead of the saved ones.
        :return:
        """
        assert action is not None
        assert noise is not None

        assert state_cur.shape == state_next.shape
        b_state = jnp.stack([state_cur, state_next], axis=0)

        # 1: State cost
        #       This function assumes a batch dimension at the end.
        assert b_state.shape == (2, 6)

        err_state = state_cur - self.goal_checker.goal_state
        assert err_state.shape == (6,)
        cost_state = 0.5 * jnp.dot(err_state, jnp.dot(self.Q, err_state))

        # 2: Control cost
        #       1/2 eRe
        cost_noise1 = 0.5 * jnp.dot(noise, jnp.dot(self.R, noise))
        cost_noise2 = jnp.dot(noise, jnp.dot(self.R, action - noise))
        cost_control = 0.5 * jnp.dot(action, jnp.dot(self.R, action))
        cost_controls = cost_noise1 + cost_noise2 + cost_control

        # 3: Collision cost.
        #       Again, this assumes a batch dimension at the end.
        collisions = self.collision_checker.check(state_cur[:, None]).squeeze(-1)
        assert collisions.shape == tuple()
        cost_collision = jnp.where(collisions, self.collision_cost, 0.0)

        # 4: CBF cost.
        bh_Vh = jax.vmap(self.get_h_)(b_state[:, :6])
        h_Vh_now, h_Vh_next = bh_Vh[0], bh_Vh[1]

        h_h_now = self.get_h_vector_(state_cur)
        h_h_next = self.get_h_vector_(state_next)

        #   Compute the violation of the discrete time CBF condition.
        h_vio = h_Vh_next - self.cbf_alpha * h_Vh_now
        h_cost_cbf = jnp.where(h_vio > 0, self.cbf_vio_cost * h_vio, self.cbf_safe_cost * h_vio)
        cost_cbf = jnp.sum(h_cost_cbf)

        # Sum all costs, reshape and return.
        if self.include_cbf_cost:
            cost = cost_state + cost_controls + cost_collision + cost_cbf
        else:
            print("not including cbf_cost!")
            cost = cost_state + cost_controls + cost_collision
        assert cost.shape == tuple()

        Vh_unsafe_thresh = -0.05
        next_unsafe_Vh = jnp.any(h_Vh_next > Vh_unsafe_thresh)

        h_vio_resample = h_h_next - self.cbf_alpha_resample * h_h_now

        violate_cbf_resample = jnp.max(h_vio_resample) > -0.05
        is_unsafe = next_unsafe_Vh | violate_cbf_resample | collisions

        info = {"h_vio": h_vio, "h_h_next": h_h_next}

        return cost[None, None], is_unsafe, info

    @partial(jax.jit, static_argnums=(0, 2))
    def evaluate_cbf_cost(
            self,
            state_cur,
            dynamics,
            state_next,
    ):
        print("re-tracing cbf cost...", end="")

        # Get the barrier function value at this state and the next (if provided)
        h_t = self.get_h_vector_(state_cur)
        h_t_plus_1 = self.get_h_vector_(state_next)

        # We want this to decrease along trajectories
        discrete_time_cbf_condition = h_t_plus_1 - self.cbf_alpha * h_t
        discrete_time_cbf_violation = np.maximum(
            discrete_time_cbf_condition, np.zeros_like(discrete_time_cbf_condition) - 0.1
        )

        if self.collision_cost is not None:
            cost = discrete_time_cbf_violation * self.collision_cost
        else:
            cost = discrete_time_cbf_violation * 1000  # default collision cost

        print("Done")

        return cost

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def evaluate_terminal_cost(
            self, state_cur, actions=None, dyna_obstacle_list=None, dynamics=None
    ):
        error_state_right = np.expand_dims(
            (state_cur - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=2
        )
        error_state_left = np.expand_dims(
            (state_cur - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=1
        )
        cost = (
                (1 / 2)
                * error_state_left
                @ np.tile(np.expand_dims(self.QN, axis=0), (state_cur.shape[1], 1, 1))
                @ error_state_right
        )
        # goal_reach = error_state_left @ error_state_right < self.goal_checker.goal_radius ** 2
        # cost += goal_reach * self.goal_cost
        return cost

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def evaluate(
            self,
            state_cur,
            actions=None,
            noises=None,
            dyna_obstacle_list=None,
            dynamics=None,
            state_next=None,
            opponent_agents=None
    ):
        return 0


class DubinsCBFCostEvaluator(QuadraticCostEvaluator):
    def __init__(
        self,
        cbf_alpha=0.9,
        goal_checker=None,
        collision_checker=None,
        Q=None,
        R=None,
        collision_cost=None,
        goal_cost=None,
    ):
        self.cbf_alpha = cbf_alpha
        QuadraticCostEvaluator.__init__(
            self, goal_checker, collision_checker, Q, R, collision_cost, goal_cost
        )
        self.cbf_alpha_resample = 0.5

    def initialize_from_config(self, config_data, section_name):
        # Get superclass parameters
        QuadraticCostEvaluator.initialize_from_config(
            self, config_data, section_name
        )
        # Get CBF-specific parameters
        self.cbf_alpha = config_data.getfloat(section_name, "cbf_alpha")
        barrier_net_section_name = config_data.get(section_name, "barrier_net")
        # self.barrier_nn = factory_from_config(
        #     barrier_net_factory_base, config_data, barrier_net_section_name
        # )
        self.include_cbf_cost = config_data.getboolean(section_name, "include_cbf_cost")
        self.cbf_vio_cost = 200.0
        self.cbf_safe_cost = 10.0
        if config_data.has_option(section_name,"include_cbf_cost"):
            self.include_cbf_cost = config_data.getboolean(section_name, "include_cbf_cost")
        else:
            self.include_cbf_cost = True

        self.obstacles = np.asarray(ast.literal_eval(config_data.get("my_collision_checker_for_collision", "obstacles")))
        self.obstacles_radius = np.asarray(
            ast.literal_eval(config_data.get("my_collision_checker_for_collision", "obstacles_radius")))

    def get_h_(self, state: jnp.ndarray) -> jnp.ndarray:
        # Get the base h value.
        h_h = self.get_h_vector_(state)
        return h_h

    def get_h_vector_(self, state: jnp.ndarray):
        obs_cfg, h_cfg = get_cfgs()
        assert state.shape == (3,)
        h_h = self.get_h_vector(h_cfg, state)
        return h_h

    def get_h_vector(self, cfg, state_cartesian):
        h_components = self.get_h_components(cfg, state_cartesian)
        h_list = list(h_components.values())
        return jnp.stack(h_list)

    def add_unsafe_eps(self, h, margin_lo: float, margin_hi: float):
        return jnp.where(h < 0, h - margin_lo, h + margin_hi)

    def get_h_components(self, cfg: ConstrCfg, state_cartesian):
        # [ vx vy wz wF wR ]
        assert state_cartesian.shape == (3,)
        # [ e_psi e_y s ]
        add_unsafe = ft.partial(self.add_unsafe_eps, margin_lo=cfg.margin_lo, margin_hi=cfg.margin_hi)

        h_track = -0.5

        h_track = self.get_h_with_obstacles(state_cartesian, h_track)

        h_track = add_unsafe(h_track)

        return {"track": h_track}

    def get_h_with_obstacles(self, state_cartesian, h):
        state_cartesian = state_cartesian[:2]
        distance_to_obstacles = jnp.linalg.norm(self.obstacles - state_cartesian, axis=1)
        obstacles_danger = self.obstacles_radius ** 2 - distance_to_obstacles ** 2
        obstacles_danger = jnp.max(obstacles_danger)
        h_with_obstacles = jnp.where(obstacles_danger > h, obstacles_danger, h)
        return h_with_obstacles

    def evaluate_cost(self, state_cur, state_next, action, noise, dynamics: AutoRallyDynamics):
        """
        :param state_cur: (nx, ) = (8, )
        :param state_next: (nx, ) = (8, )
        :param action: (nu, ) = (2, )
        :param noise: (nu, ) = (2, )
        :param dynamics:
        :param ncbf_weights: Weights for the NCBF to use instead of the saved ones.
        :return:
        """
        assert action is not None
        assert noise is not None
        assert state_cur.shape == state_next.shape

        b_state = jnp.stack([state_cur, state_next], axis=0)

        # 1: State cost
        #       This function assumes a batch dimension at the end.
        assert b_state.shape == (2, 3)

        err_state = state_cur - self.goal_checker.goal_state
        assert err_state.shape == (3,)
        cost_state = 0.5 * jnp.dot(err_state, jnp.dot(self.Q, err_state))

        # 2: Control cost
        #       1/2 eRe
        cost_noise1 = 0.5 * jnp.dot(noise, jnp.dot(self.R, noise))
        cost_noise2 = jnp.dot(noise, jnp.dot(self.R, action - noise))
        cost_control = 0.5 * jnp.dot(action, jnp.dot(self.R, action))
        cost_controls = cost_noise1 + cost_noise2 + cost_control

        # 3: Collision cost.
        #       Again, this assumes a batch dimension at the end.
        collisions = self.collision_checker.check(state_cur[:, None]).squeeze(-1)
        assert collisions.shape == tuple()
        cost_collision = jnp.where(collisions, self.collision_cost, 0.0)

        # 4: CBF cost.
        bh_Vh = jax.vmap(self.get_h_)(b_state[:, :3])
        h_Vh_now, h_Vh_next = bh_Vh[0], bh_Vh[1]

        h_h_now = self.get_h_vector_(state_cur)
        h_h_next = self.get_h_vector_(state_next)

        #   Compute the violation of the discrete time CBF condition.
        h_vio = h_Vh_next - self.cbf_alpha * h_Vh_now
        h_cost_cbf = jnp.where(h_vio > 0, self.cbf_vio_cost * h_vio, self.cbf_safe_cost * h_vio)
        cost_cbf = jnp.sum(h_cost_cbf)

        # Sum all costs, reshape and return.
        if self.include_cbf_cost:
            cost = cost_state + cost_controls + cost_collision + cost_cbf
        else:
            print("not including cbf_cost!")
            cost = cost_state + cost_controls + cost_collision
        assert cost.shape == tuple()

        Vh_unsafe_thresh = -0.05
        next_unsafe_Vh = jnp.any(h_Vh_next > Vh_unsafe_thresh)

        h_vio_resample = h_h_next - self.cbf_alpha_resample * h_h_now

        violate_cbf_resample = jnp.max(h_vio_resample) > -0.05
        is_unsafe = next_unsafe_Vh | violate_cbf_resample | collisions

        info = {"h_vio": h_vio, "h_h_next": h_h_next}

        return cost[None, None], is_unsafe, info

    @partial(jax.jit, static_argnums=(0, 2))
    def evaluate_cbf_cost(
            self,
            state_cur,
            dynamics,
            state_next,
    ):
        print("re-tracing cbf cost...", end="")

        # Get the barrier function value at this state and the next (if provided)
        h_t = self.get_h_vector_(state_cur)
        h_t_plus_1 = self.get_h_vector_(state_next)

        # We want this to decrease along trajectories
        discrete_time_cbf_condition = h_t_plus_1 - self.cbf_alpha * h_t
        discrete_time_cbf_violation = np.maximum(
            discrete_time_cbf_condition, np.zeros_like(discrete_time_cbf_condition) - 0.1
        )

        if self.collision_cost is not None:
            cost = discrete_time_cbf_violation * self.collision_cost
        else:
            cost = discrete_time_cbf_violation * 1000  # default collision cost

        print("Done")

        return cost

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def evaluate_terminal_cost(
            self, state_cur, actions=None, dyna_obstacle_list=None, dynamics=None
    ):
        error_state_right = np.expand_dims(
            (state_cur - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=2
        )
        error_state_left = np.expand_dims(
            (state_cur - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=1
        )
        cost = (
                (1 / 2)
                * error_state_left
                @ np.tile(np.expand_dims(self.QN, axis=0), (state_cur.shape[1], 1, 1))
                @ error_state_right
        )
        # goal_reach = error_state_left @ error_state_right < self.goal_checker.goal_radius ** 2
        # cost += goal_reach * self.goal_cost
        return cost

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def evaluate(
            self,
            state_cur,
            actions=None,
            noises=None,
            dyna_obstacle_list=None,
            dynamics=None,
            state_next=None,
            opponent_agents=None
    ):
        return 0



class AutorallyMPPICostEvaluator(QuadraticCostEvaluator):
    def __init__(
        self,
        goal_checker=None,
        collision_checker=None,
        Q=None,
        R=None,
        collision_cost=None,
        goal_cost=None,
    ):
        QuadraticCostEvaluator.__init__(
            self, goal_checker, collision_checker, Q, R, collision_cost, goal_cost
        )

    def initialize_from_config(self, config_data, section_name):
        QuadraticCostEvaluator.initialize_from_config(self, config_data, section_name)

        # Get CBF-specific parameters
        if config_data.has_option("logger", "ablation"):
            if config_data.get("logger", "ablation") == "mppi_plus_local_repair":
                self.cbf_alpha = config_data.getfloat(section_name, "cbf_alpha")
                barrier_net_section_name = config_data.get(section_name, "barrier_net")
                self.barrier_nn = factory_from_config(
                    barrier_net_factory_base, config_data, barrier_net_section_name
                )

    @partial(jax.jit, static_argnums=(0, 2))
    def evaluate_cbf_cost(
            self,
            state_cur,
            dynamics,
            state_next,
    ):
        print("re-tracing cbf cost...", end="")
        map_state = self.global_to_local_coordinate_transform(state_cur, dynamics)

        # Get the barrier function value at this state and the next (if provided)
        h_t = self.barrier_nn(map_state.T, state_cur)

        map_state_next = self.global_to_local_coordinate_transform(
            state_next, dynamics
        )
        h_t_plus_1 = self.barrier_nn(map_state_next.T, state_cur)

        # We want this to decrease along trajectories
        discrete_time_cbf_condition = h_t_plus_1 - self.cbf_alpha * h_t
        discrete_time_cbf_violation = np.maximum(
            discrete_time_cbf_condition, np.zeros_like(discrete_time_cbf_condition) - 0.1
        )

        if self.collision_cost is not None:
            cost = discrete_time_cbf_violation * self.collision_cost
        else:
            cost = discrete_time_cbf_violation * 1000  # default collision cost

        print("Done")

        return cost

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def evaluate(
        self,
        state_cur,
        actions=None,
        noises=None,
        dyna_obstacle_list=None,
        dynamics=None,
        opponent_agents=None
    ):
        map_state = self.global_to_local_coordinate_transform(state_cur, dynamics)
        error_state_right = np.expand_dims(
            (map_state - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=2
        )
        error_state_left = np.expand_dims(
            (map_state - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=1
        )
        # 1/2 xQx
        cost = (
            (1 / 2)
            * error_state_left
            @ np.tile(np.expand_dims(self.Q, axis=0), (state_cur.shape[1], 1, 1))
            @ error_state_right
        )
        if actions is not None:
            actions_left = np.expand_dims(actions.T, axis=1)
            actions_right = np.expand_dims(actions.T, axis=2)
            if noises is not None:
                noises_left = np.expand_dims(noises.T, axis=1)
                noises_right = np.expand_dims(noises.T, axis=2)
                # 1/2 eRe
                cost += (
                    1
                    / 2
                    * noises_left
                    @ np.tile(
                        np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1)
                    )
                    @ noises_right
                )
                # vRe
                cost += (
                    (actions_left - noises_left)
                    @ np.tile(
                        np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1)
                    )
                    @ noises_right
                )

            # 1/2 uRu
            cost += (
                (1 / 2)
                * actions_left
                @ np.tile(np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1))
                @ actions_right
            )

        collisions = self.collision_checker.check(
            map_state, state_cur, opponent_agents=opponent_agents
        )
        collisions = collisions.reshape((-1, 1, 1))
        if self.collision_cost is not None:
            cost += collisions * self.collision_cost
        else:
            cost += collisions * 1000  # default collision cost

        return cost

    def evaluate_vv(
        self,
        state_cur,
        actions=None,
        noises=None,
        dyna_obstacle_list=None,
        dynamics=None,
    ):
        # FIXME temp
        state_cur = state_cur.reshape(state_cur.shape[0], -1)
        actions = actions.reshape(actions.shape[0], -1)
        noises = noises.reshape(noises.shape[0], -1)

        map_state = self.global_to_local_coordinate_transform(state_cur, dynamics)
        error_state_right = np.expand_dims(
            (map_state - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=2
        )
        error_state_left = np.expand_dims(
            (map_state - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=1
        )
        xQx = (
            (1 / 2)
            * error_state_left
            @ np.tile(np.expand_dims(self.Q, axis=0), (state_cur.shape[1], 1, 1))
            @ error_state_right
        )
        cost = xQx.copy()
        if actions is not None:
            actions_left = np.expand_dims(actions.T, axis=1)
            actions_right = np.expand_dims(actions.T, axis=2)
            if noises is not None:
                noises_left = np.expand_dims(noises.T, axis=1)
                noises_right = np.expand_dims(noises.T, axis=2)
                uRu = (
                    1
                    / 2
                    * noises_left
                    @ np.tile(
                        np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1)
                    )
                    @ noises_right
                )
                cost += uRu
                vRe = (
                    (actions_left - noises_left)
                    @ np.tile(
                        np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1)
                    )
                    @ noises_right
                )
                cost += vRe

            eRe = (
                (1 / 2)
                * actions_left
                @ np.tile(np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1))
                @ actions_right
            )
            cost += eRe

        return cost, xQx, uRu, vRe, eRe, map_state[5], map_state[6], map_state[7]

    def evaluate_terminal_cost(
        self, state_cur, actions=None, dyna_obstacle_list=None, dynamics=None
    ):
        map_state = self.global_to_local_coordinate_transform(state_cur, dynamics)
        error_state_right = np.expand_dims(
            (map_state - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=2
        )
        error_state_left = np.expand_dims(
            (map_state - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=1
        )
        cost = (
            (1 / 2)
            * error_state_left
            @ np.tile(np.expand_dims(self.QN, axis=0), (state_cur.shape[1], 1, 1))
            @ error_state_right
        )
        # collisions = self.collision_checker.check(state_cur)  # True for collision, False for no collision
        # collisions = collisions.reshape((-1, 1, 1))
        # if self.collision_cost is not None:
        #     cost += collisions * self.collision_cost
        # else:
        #     cost += collisions * 1000  # default collision cost
        return cost

    def global_to_curvi(self, state, dynamics):
        track: MapCA = dynamics.track
        assert state.ndim == 1

        pos, psi = state[-2:], state[-3]
        epsi, ey, s = track.localize_lerp(pos, psi)
        curvi = np.array([epsi, ey, s])
        return curvi

    def global_to_local_coordinate_transform(self, state, dynamics):
        track: MapCA = dynamics.track
        assert state.ndim == 2
        xb_state = state
        bx_state = xb_state.T
        b_pos = xb_state[-2:, :].T
        b_psi = xb_state[-3, :]

        b, nx = bx_state.shape

        b_epsi, b_ey, b_s = jax.vmap(track.localize_lerp)(b_pos, b_psi)

        # (3, b)
        xb_curvilinear = np.stack([b_epsi, b_ey, b_s], axis=0)
        assert xb_curvilinear.shape == (3, b)

        new_state = state.at[-3:, :].set(xb_curvilinear)
        # e_psi, e_y, s = track.localize(
        #     np.array((state[-2, :], state[-1, :])), state[-3, :]
        # )
        # new_state = state.copy()
        # new_state = new_state.at[-3:, :].set(np.vstack((e_psi, e_y, s)))

        assert new_state.shape == state.shape
        return new_state

class AutorallyMPPICBFCostEvaluator(AutorallyMPPICostEvaluator):
    def __init__(
        self,
        cbf_alpha=0.9,
        goal_checker=None,
        collision_checker=None,
        Q=None,
        R=None,
        collision_cost=None,
        goal_cost=None,
    ):
        self.cbf_alpha = cbf_alpha
        AutorallyMPPICostEvaluator.__init__(
            self, goal_checker, collision_checker, Q, R, collision_cost, goal_cost
        )
        self.cbf_alpha = 0.9
        self.cbf_alpha_resample = 0.5
        # self.cbf_alpha_resample = 0.85
        # self.cbf_vio_cost = 1_000.0
        self.cbf_vio_cost = 2_000.0
        self.cbf_safe_cost = 100.0

    def initialize_from_config(self, config_data, section_name):
        # Get superclass parameters
        AutorallyMPPICostEvaluator.initialize_from_config(
            self, config_data, section_name
        )
        # Get CBF-specific parameters
        self.cbf_alpha = config_data.getfloat(section_name, "cbf_alpha")
        barrier_net_section_name = config_data.get(section_name, "barrier_net")
        self.barrier_nn = factory_from_config(
            barrier_net_factory_base, config_data, barrier_net_section_name
        )
        if config_data.has_option(section_name,"include_cbf_cost"):
            self.include_cbf_cost = config_data.getboolean(section_name, "include_cbf_cost")
        else:
            self.include_cbf_cost = True
        self.obstacles = np.asarray(ast.literal_eval(config_data.get("my_collision_checker_for_collision", "obstacles")))
        self.obstacles_radius = np.asarray(
            ast.literal_eval(config_data.get("my_collision_checker_for_collision", "obstacles_radius")))


    def get_h_(self, state: jnp.ndarray, map_state: jnp.ndarray) -> jnp.ndarray:
        # Get the base h value.
        h_h = self.get_h_vector_(state, map_state)
        return h_h

    def get_h_vector_(self, state: jnp.ndarray, map_state: jnp.ndarray):
        obs_cfg, h_cfg = get_cfgs()
        assert state.shape == (8,)
        assert map_state.shape == (3,)
        h_h = self.get_h_vector(h_cfg, state, map_state)
        return h_h

    def get_h_vector(self, cfg, state_cartesian, curvilinear):
        h_components = self.get_h_components(cfg, state_cartesian, curvilinear)
        h_list = list(h_components.values())
        return jnp.stack(h_list)

    def add_unsafe_eps(self, h, margin_lo: float, margin_hi: float):
        return jnp.where(h < 0, h - margin_lo, h + margin_hi)

    def get_h_components(self, cfg: ConstrCfg, state_cartesian, curvilinear):
        # [ vx vy wz wF wR ]
        assert state_cartesian.shape == (8,)
        # [ e_psi e_y s ]
        assert curvilinear.shape == (3,)
        add_unsafe = ft.partial(self.add_unsafe_eps, margin_lo=cfg.margin_lo, margin_hi=cfg.margin_hi)

        e_psi, e_y, s = curvilinear

        is_term = jnp.abs(e_y) >= cfg.track_width_term  # True if crash

        h_track = e_y ** 2 - cfg.track_width ** 2  # h_track > 0 if collision

        h_track = self.get_h_with_obstacles(state_cartesian, h_track)

        h_track = add_unsafe(h_track)
        h_track = jnp.where(is_term, cfg.h_term + cfg.margin_hi, h_track)

        return {"track": h_track}

    def get_h_with_obstacles(self, state_cartesian, h):
        state_cartesian = state_cartesian[6:8]
        distance_to_obstacles = jnp.linalg.norm(self.obstacles - state_cartesian, axis=1)
        obstacles_danger = self.obstacles_radius ** 2 - distance_to_obstacles ** 2
        obstacles_danger = jnp.max(obstacles_danger)
        h_with_obstacles = jnp.where(obstacles_danger > h, obstacles_danger, h)
        return h_with_obstacles


    def evaluate_cost(self, state_cur, state_next, action, noise, dynamics: AutoRallyDynamics):
        """
        :param state_cur: (nx, ) = (8, )
        :param state_next: (nx, ) = (8, )
        :param action: (nu, ) = (2, )
        :param noise: (nu, ) = (2, )
        :param dynamics:
        :param ncbf_weights: Weights for the NCBF to use instead of the saved ones..
        :return:
        """
        assert action is not None
        assert noise is not None

        assert state_cur.shape == state_next.shape
        b_state = jnp.stack([state_cur, state_next], axis=0)

        to_local = ft.partial(self.global_to_local_coordinate_transform, dynamics=dynamics)

        # 1: State cost
        #       This function assumes a batch dimension at the end.
        b_map_state = jax.vmap(to_local)(b_state[:, :, None]).squeeze(-1)
        assert b_map_state.shape == (2, 8)

        map_state = b_map_state[0]
        err_state = map_state - self.goal_checker.goal_state
        assert err_state.shape == (8,)
        cost_state = 0.5 * jnp.dot(err_state, jnp.dot(self.Q, err_state))

        # 2: Control cost
        #       1/2 eRe
        cost_noise1 = 0.5 * jnp.dot(noise, jnp.dot(self.R, noise))
        cost_noise2 = jnp.dot(noise, jnp.dot(self.R, action - noise))
        cost_control = 0.5 * jnp.dot(action, jnp.dot(self.R, action))
        cost_controls = cost_noise1 + cost_noise2 + cost_control

        # 3: Collision cost.
        #       Again, this assumes a batch dimension at the end.
        collisions = self.collision_checker.check(map_state[:, None], state_cur[:, None]).squeeze(-1)
        assert collisions.shape == tuple()
        cost_collision = jnp.where(collisions, self.collision_cost, 0.0)

        # 4: CBF cost.
        bh_Vh = jax.vmap(self.get_h_)(b_state[:, :8], b_map_state[:, -3:])
        h_Vh_now, h_Vh_next = bh_Vh[0], bh_Vh[1]

        h_h_now = self.get_h_vector_(state_cur, b_map_state[0, -3:])
        h_h_next = self.get_h_vector_(state_next, b_map_state[1, -3:])

        #   Compute the violation of the discrete time CBF condition.
        #   \dot{V} + λ V <= 0.
        #   =>  V(x_{t + Δ}) <= exp(-λ Δ) V(x_t).
        #   => V(x_{t + Δ}) - exp(-λ Δ) V(x_t) <= 0.
        #   => V(x_{t + Δ}) - α V(x_t) <= 0.
        #   λ=0 => α=1,    λ=∞ => α=0
        #   α=1 is the most conservative, α=0 is the least conservative.
        h_vio = h_Vh_next - self.cbf_alpha * h_Vh_now
        h_cost_cbf = jnp.where(h_vio > 0, self.cbf_vio_cost * h_vio, self.cbf_safe_cost * h_vio)
        # #       If we are unsafe now, then maximize safety of the next step.
        # is_unsafe = jnp.any(h_Vh_now > 0)
        # h_cost_cbf_unsafe = self.cbf_vio_cost * h_Vh_next
        # h_cost_cbf = jnp.where(is_unsafe, h_cost_cbf_unsafe, h_cost_cbf)
        cost_cbf = jnp.sum(h_cost_cbf)

        # Sum all costs, reshape and return.
        if self.include_cbf_cost:
            cost = cost_state + cost_controls + cost_collision + cost_cbf
        else:
            print("not including cbf_cost!")
            cost = cost_state + cost_controls + cost_collision
        assert cost.shape == tuple()

        Vh_unsafe_thresh = -0.05
        next_unsafe_Vh = jnp.any(h_Vh_next > Vh_unsafe_thresh)
        # h_vio_resample = h_Vh_now - self.cbf_alpha_resample * h_Vh_next
        h_vio_resample = h_h_next - self.cbf_alpha_resample * h_h_now
        # h_vio_resample = h_h_now - self.cbf_alpha_resample * h_h_next # TODO: This should be h_h_next - self.cbf_alpha_resample * h_h_now

        violate_cbf_resample = jnp.max(h_vio_resample) > -0.05
        is_unsafe = next_unsafe_Vh | violate_cbf_resample | collisions

        # is_unsafe = next_unsafe_Vh | collisions
        # is_unsafe = jnp.any(h_vio > 0) | collisions
        # is_unsafe = collisions

        # info = {"h_vio": h_vio, "h_Vh_next": h_Vh_next, "h_h_next": h_h_next}
        info = {"h_vio": h_vio, "h_h_curr": h_h_now, "h_h_next": h_h_next, "h_Vh_curr": h_Vh_now, "h_Vh_next": h_Vh_next}

        return cost[None, None], is_unsafe, info

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def evaluate(
        self,
        state_cur,
        actions=None,
        noises=None,
        dyna_obstacle_list=None,
        dynamics=None,
        state_next=None,
        opponent_agents=None
    ):
        print("re-tracing evaluate!")
        # This is all the same as AutorallyMPPICostEvaluator.evaluate
        # except that we don't apply the collision cost
        # (we use the CBF cost instead of a collision cost, but that's applied in a
        # different function)
        map_state = self.global_to_local_coordinate_transform(state_cur, dynamics)
        # print(map_state.shape)
        error_state_right = np.expand_dims(
            (map_state - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=2
        )
        error_state_left = np.expand_dims(
            (map_state - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=1
        )
        # 1/2 xQx
        cost = (
            (1 / 2)
            * error_state_left
            @ np.tile(np.expand_dims(self.Q, axis=0), (state_cur.shape[1], 1, 1))
            @ error_state_right
        )
        if actions is not None:
            actions_left = np.expand_dims(actions.T, axis=1)
            actions_right = np.expand_dims(actions.T, axis=2)
            if noises is not None:
                noises_left = np.expand_dims(noises.T, axis=1)
                noises_right = np.expand_dims(noises.T, axis=2)
                # 1/2 eRe
                cost += (
                    1
                    / 2
                    * noises_left
                    @ np.tile(
                        np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1)
                    )
                    @ noises_right
                )
                # vRe
                cost += (
                    (actions_left - noises_left)
                    @ np.tile(
                        np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1)
                    )
                    @ noises_right
                )

            # 1/2 uRu
            cost += (
                (1 / 2)
                * actions_left
                @ np.tile(np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1))
                @ actions_right
            )

        # Get the barrier function value at this state and the next (if provided)
        h_t = self.barrier_nn(map_state.T, state_cur)

        if state_next is not None:
            map_state_next = self.global_to_local_coordinate_transform(
                state_next, dynamics
            )
            h_t_plus_1 = self.barrier_nn(map_state_next.T, state_next)
        else:
            h_t_plus_1 = h_t

        # We want this to decrease along trajectories
        discrete_time_cbf_condition = h_t_plus_1 - self.cbf_alpha * h_t
        discrete_time_cbf_violation = np.maximum(
            discrete_time_cbf_condition, np.zeros_like(discrete_time_cbf_condition) - 0.1
        ).reshape(-1, 1, 1)

        if self.include_cbf_cost:
            cost += discrete_time_cbf_violation * 1000  # default collision cost

        # # Also consider collisions in addition to the CBF
        collisions = self.collision_checker.check(
            map_state, state_cur, opponent_agents=opponent_agents
        )
        collisions = collisions.reshape((-1, 1, 1))
        if self.collision_cost is not None:
            cost += collisions * self.collision_cost
        else:
            cost += collisions * 1000  # default collision cost
        return cost

    @partial(jax.jit, static_argnums=(0, 2))
    def evaluate_cbf_cost(
        self,
        state_cur,
        dynamics,
        state_next,
    ):
        print("re-tracing cbf cost...", end="")
        map_state = self.global_to_local_coordinate_transform(state_cur, dynamics)

        # Get the barrier function value at this state and the next (if provided)
        h_t = self.barrier_nn(map_state.T, state_cur)

        map_state_next = self.global_to_local_coordinate_transform(
            state_next, dynamics
        )
        h_t_plus_1 = self.barrier_nn(map_state_next.T, state_cur)

        # We want this to decrease along trajectories
        discrete_time_cbf_condition = h_t_plus_1 - self.cbf_alpha * h_t
        discrete_time_cbf_violation = np.maximum(
            discrete_time_cbf_condition, np.zeros_like(discrete_time_cbf_condition) - 0.1
        )

        if self.collision_cost is not None:
            cost = discrete_time_cbf_violation * self.collision_cost
        else:
            cost = discrete_time_cbf_violation * 1000  # default collision cost

        print("Done")

        return cost


class TerminalCostEvaluator(CostEvaluator):
    def __init__(
        self,
        goal_checker=None,
        collision_checker=None,
        Q=None,
        R=None,
        collision_cost=None,
        goal_cost=None,
    ):
        CostEvaluator.__init__(self, goal_checker, collision_checker)
        self.collision_cost = collision_cost
        self.goal_cost = goal_cost
        self.dense = None

    def initialize_from_config(self, config_data, section_name):
        if config_data.has_option(section_name, "collision_cost"):
            self.collision_cost = config_data.getfloat(
                section_name, "collision_cost"
            )  # collision_cost should be positive
        if config_data.has_option(section_name, "goal_cost"):
            self.goal_cost = config_data.getfloat(
                section_name, "goal_cost"
            )  # goal_cost should be negative
        if config_data.has_option(section_name, "goal_checker"):
            goal_checker_section_name = config_data.get(section_name, "goal_checker")
            self.goal_checker = factory_from_config(
                goal_checker_factory_base, config_data, goal_checker_section_name
            )
        if config_data.has_option(section_name, "collision_checker"):
            collision_checker_section_name = config_data.get(
                section_name, "collision_checker"
            )
            self.collision_checker = factory_from_config(
                collision_checker_factory_base,
                config_data,
                collision_checker_section_name,
            )
        if config_data.has_option(section_name, "dense"):
            self.dense = config_data.getboolean(section_name, "dense")

    def evaluate(self, state_cur, actions=None, dyna_obstacle_list=None, opponent_agents=None):
        cost = 0
        if self.collision_checker.check(
            state_cur
        ):  # True for collision, False for no collision
            if self.collision_cost is not None:
                cost += self.collision_cost
            else:
                cost += 1000  # default collision cost
        if self.dense:
            cost += 10 * self.goal_checker.dist(state_cur)

        if self.goal_checker.check(
            state_cur
        ):  # True for goal reached, False for goal not reached
            if self.goal_cost is not None:
                cost += self.goal_cost
            else:
                cost += -5000  # default goal cost

        return cost


class AbstractCostEvaluator(CostEvaluator):
    def __init__(
        self,
        goal_checker=None,
        sub_goal_checker=None,
        collision_checker=None,
        non_achievable_cost=None,
        achievable_cost=None,
    ):
        CostEvaluator.__init__(self, goal_checker, collision_checker)
        self.non_achievable_cost = non_achievable_cost
        self.achievable_cost = achievable_cost
        self.dense = None
        self.subgoal_infeasible_cost = None
        self.sub_goal_checker = sub_goal_checker

    def initialize_from_config(self, config_data, section_name):
        goal_checker_section_name = config_data.get(section_name, "goal_checker")
        self.goal_checker = factory_from_config(
            goal_checker_factory_base, config_data, goal_checker_section_name
        )
        self.non_achievable_cost = config_data.getfloat(
            section_name, "non_achievable_cost"
        )
        self.ultimate_goal_cost = config_data.getfloat(
            section_name, "ultimate_goal_cost"
        )
        if config_data.has_option(section_name, "achievable_cost"):
            self.achievable_cost = config_data.getfloat(section_name, "achievable_cost")
        else:
            self.achievable_cost = -self.non_achievable_cost
        if config_data.has_option(section_name, "subgoal_infeasible_cost"):
            self.subgoal_infeasible_cost = config_data.getfloat(
                section_name, "subgoal_infeasible_cost"
            )

    def set_sub_goal_checker(self, sub_goal_checker):
        self.sub_goal_checker = sub_goal_checker

    def evaluate(self, state_cur, state_next=None, action=None):
        assert (
            np.linalg.norm(
                action.reshape(self.sub_goal_checker.goal_state.shape)
                - self.sub_goal_checker.goal_state
            )
            < 1e-5
        )
        assert state_next is not None
        cost = 0
        if self.sub_goal_checker.check(state_next):
            cost += self.achievable_cost
        else:
            cost += self.non_achievable_cost
        if self.goal_checker.check(state_next):
            cost += self.ultimate_goal_cost
        if (
            self.collision_checker.check(self.sub_goal_checker.goal_state)
            and self.subgoal_infeasible_cost is not None
        ):
            cost += self.subgoal_infeasible_cost
        return cost


class MPPICostEvaluator(QuadraticCostEvaluator):
    def __init__(
        self,
        goal_checker=None,
        collision_checker=None,
        Q=None,
        R=None,
        collision_cost=None,
        goal_cost=None,
    ):
        QuadraticCostEvaluator.__init__(
            self, goal_checker, collision_checker, Q, R, collision_cost, goal_cost
        )

    def initialize_from_config(self, config_data, section_name):
        QuadraticCostEvaluator.initialize_from_config(self, config_data, section_name)

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def evaluate(
        self,
        state_cur,
        actions=None,
        noises=None,
        dyna_obstacle_list=None,
        dynamics=None,
        opponent_agents=None
    ):
        print("re-tracing evaluate!")
        error_state_right = np.expand_dims(
            (state_cur - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=2
        )
        error_state_left = np.expand_dims(
            (state_cur - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=1
        )
        cost = (
            (1 / 2)
            * error_state_left
            @ np.tile(np.expand_dims(self.Q, axis=0), (state_cur.shape[1], 1, 1))
            @ error_state_right
        )
        if actions is not None:
            actions_left = np.expand_dims(actions.T, axis=1)
            actions_right = np.expand_dims(actions.T, axis=2)
            if noises is not None:
                noises_left = np.expand_dims(noises.T, axis=1)
                noises_right = np.expand_dims(noises.T, axis=2)
                cost += (
                    1
                    / 2
                    * noises_left
                    @ np.tile(
                        np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1)
                    )
                    @ noises_right
                )
                cost += (
                    (actions_left - noises_left)
                    @ np.tile(
                        np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1)
                    )
                    @ noises_right
                )
            cost += (
                (1 / 2)
                * actions_left
                @ np.tile(np.expand_dims(self.R, axis=0), (state_cur.shape[1], 1, 1))
                @ actions_right
            )
        if self.collision_checker is not None:
            # vectorized states collision cost evaluation
            collisions = self.collision_checker.check(
                state_cur, opponent_agents=opponent_agents
            )  # True for collision, False for no collision
            collisions = collisions.reshape((-1, 1, 1))
            if self.collision_cost is not None:
                cost += collisions * self.collision_cost
            else:
                cost += collisions * 1000  # default collision cost

        return cost
    @partial(jax.jit, static_argnums=(0, 3, 4))
    def evaluate_terminal_cost(
        self, state_cur, actions=None, dyna_obstacle_list=None, dynamics=None
    ):
        print("re-tracing terminal cost evaluate!")
        error_state_right = np.expand_dims(
            (state_cur - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=2
        )
        error_state_left = np.expand_dims(
            (state_cur - self.goal_checker.goal_state.reshape((-1, 1))).T, axis=1
        )
        cost = (
            (1 / 2)
            * error_state_left
            @ np.tile(np.expand_dims(self.QN, axis=0), (state_cur.shape[1], 1, 1))
            @ error_state_right
        )
        # collisions = self.collision_checker.check(state_cur)  # True for collision, False for no collision
        # collisions = collisions.reshape((-1, 1, 1))
        # if self.collision_cost is not None:
        #     cost += collisions * self.collision_cost
        # else:
        #     cost += collisions * 1000  # default collision cost
        return cost


