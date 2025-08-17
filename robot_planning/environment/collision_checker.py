import ipdb
import jax.numpy as np
import jax
import ast
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import kinematics_factory_base
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.environment.kinematics.simulated_kinematics import PointKinematics
from robot_planning.environment.kinematics.simulated_kinematics import (
    BicycleModelKinematics,
)


class CollisionChecker(object):
    def __init__(self, obstacles=None, kinematics=None, field_boundary=None):
        self.kinematics = kinematics  # for more complicated collision checkers
        self.obstacles = obstacles
        self.field_boundary = field_boundary

    def initialize_from_config(self, config_data, section_name):
        pass

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def set_obstacles_radius(self, obstacles_radius):
        self.obstacles_radius = obstacles_radius

    def set_other_agents_list(self, other_agents_list):
        self.other_agents_list = other_agents_list

    def get_obstacle_list(self):
        return np.concatenate((self.obstacles, self.obstacles_radius[np.newaxis, :].T), axis=1)

    def check(self, state_cur):
        raise NotImplementedError


class PointCollisionChecker(CollisionChecker):
    def __init__(self, obstacles=None, kinematics=None):
        CollisionChecker.__init__(self, obstacles, kinematics)

    def initialize_from_config(self, config_data, section_name):
        kinematics_section_name = config_data.get(section_name, "kinematics")
        self.kinematics = factory_from_config(
            kinematics_factory_base, config_data, kinematics_section_name
        )
        assert isinstance(
            self.kinematics, PointKinematics
        ), "The PointCollisionChecker should have PointKinematics"
        if config_data.has_option(section_name, "obstacles"):
            self.obstacles = np.asarray(
                ast.literal_eval(config_data.get(section_name, "obstacles"))
            )
            self.obstacles_radius = np.asarray(
                ast.literal_eval(config_data.get(section_name, "obstacles_radius"))
            )
            if len(self.obstacles) is not len(self.obstacles_radius):
                raise ValueError("the numbers of obstacles and radii do not match")
        if config_data.has_option(section_name, "field_boundary"):
            self.field_boundary = np.asarray(
                ast.literal_eval(config_data.get(section_name, "field_boundary"))
            )
        self.other_agents_list = None

    def get_obstacle_list(self):
        return np.concatenate((self.obstacles, self.obstacles_radius[np.newaxis, :].T), axis=1)

    def check(self, state_cur, opponent_agents=None, check_other_agents=None):  # True for collision, False for no collision
        obstacle_collisions = self.check_collision_with_obstacles(state_cur)
        return obstacle_collisions
        # return self.check_collisions_with_boundaries(state_cur)

    def check_collision_with_obstacles(self, cartesian_state_cur):
        if cartesian_state_cur.ndim == 1:
            if (np.linalg.norm(self.obstacles - cartesian_state_cur[:2], axis=1) < (self.obstacles_radius[np.newaxis, :] + self.kinematics.get_radius())).any():
                # if the vehicle collide with any one of the obstacles
                return True
            return False
        else:
            cartesian_state_cur = cartesian_state_cur.T
            obstacles_reshaped = self.obstacles[np.newaxis, :, :]
            cartesian_state_reshaped = cartesian_state_cur[:, np.newaxis, :2]

            distance_to_obstacles = np.linalg.norm(cartesian_state_reshaped - obstacles_reshaped, axis=2) # of shape (state_num, obstacle_num)
            collisions = (distance_to_obstacles < (self.obstacles_radius[np.newaxis, :] + self.kinematics.get_radius()) ).any(axis=1) # if a state's distance to any obstacle is less than radius, it collides
            return collisions

class BicycleModelCollisionChecker(CollisionChecker):
    def __init__(self, obstacles=None, kinematics=None):
        CollisionChecker.__init__(self, obstacles, kinematics)

    def initialize_from_config(self, config_data, section_name):
        kinematics_section_name = config_data.get(section_name, "kinematics")
        self.kinematics = factory_from_config(
            kinematics_factory_base, config_data, kinematics_section_name
        )
        assert isinstance(
            self.kinematics, BicycleModelKinematics
        ), "The BicycleModelCollisionChecker should have BicycleModelKinematics"
        if config_data.has_option(section_name, "obstacles"):
            self.obstacles = np.asarray(
                ast.literal_eval(config_data.get(section_name, "obstacles"))
            )
        if config_data.has_option(section_name, "obstacles_radius"):
            self.obstacles_radius = np.asarray(
                ast.literal_eval(config_data.get(section_name, "obstacles_radius"))
            )
        if config_data.has_option(section_name, "agent_safety_distance"):
            self.agent_safety_distance = config_data.getfloat(
                section_name, "agent_safety_distance"
            )
        self.other_agents_list = None

    def check(
        self, state_cur, opponent_agents=None):  # True for collision, False for no collision
        state_cur = np.squeeze(state_cur)
        vertex_list = self.kinematics.compute_rectangle_vertices_from_state(state_cur)
        for vertex in vertex_list:
            for i in range(len(self.obstacles)):
                if (
                    np.linalg.norm(self.obstacles[i] - vertex)
                    < self.obstacles_radius[i]
                ):
                    return True
        if self.other_agents_list is not None:
            for agent in self.other_agents_list:
                if (
                    np.linalg.norm(agent.state[:2] - state_cur[:2])
                    < self.agent_safety_distance
                ):  # TODO: this is hack. Need to use a generic representation for agent kinematics
                    return True
        return False


class AutorallyCollisionChecker(PointCollisionChecker):
    def __init__(self, obstacles=None, kinematics=None):
        CollisionChecker.__init__(self, obstacles, kinematics)

    def initialize_from_config(self, config_data, section_name):
        PointCollisionChecker.initialize_from_config(self, config_data, section_name)
        self.track_width = config_data.getfloat(section_name, "track_width")
        kinematics_section_name = config_data.get(section_name, "kinematics")
        self.kinematics = factory_from_config(
            kinematics_factory_base, config_data, kinematics_section_name
        )
        if config_data.has_option(section_name, "obstacles"):
            self.obstacles = np.asarray(ast.literal_eval(config_data.get(section_name, "obstacles")))
            self.obstacles_radius = np.asarray(ast.literal_eval(config_data.get(section_name, "obstacles_radius")))
        # dynamics_section_name = config_data.get(section_name, 'dynamics')
        # self.dynamics = factory_from_config(dynamics_factory_base, config_data, dynamics_section_name)

    def check(self, state_cur, cartesian_state_cur=None, opponent_agents=None):
        if cartesian_state_cur is None:
            return self.check_collision_with_boundaries(state_cur)
        else:
            boundary_collisions = self.check_collision_with_boundaries(state_cur)
            obstacle_collisions = self.check_collision_with_obstacles(cartesian_state_cur)
            # if the state collided with either a boundary or an obstacle, it returns True
            return boundary_collisions | obstacle_collisions

    def check_collision_with_boundaries(self, map_state):
        is_oob = (map_state[-2] < -self.track_width) | (self.track_width < map_state[-2])
        if map_state.ndim == 1:
            # Single state. (8, )
            return is_oob
        else:
            # Trajectory. (8, T)
            collisions = np.where(is_oob, 1, 0)
            return collisions

    def check_collision_with_obstacles(self, cartesian_state_cur):
        if cartesian_state_cur.ndim == 1:
            if (np.linalg.norm(self.obstacles - cartesian_state_cur[6:8], axis=1) < self.obstacles_radius[np.newaxis, :]).any():
                # if the vehicle collide with any one of the obstacles
                return True
            return False
        else:
            cartesian_state_cur = cartesian_state_cur.T
            obstacles_reshaped = self.obstacles[np.newaxis, :, :]
            cartesian_state_reshaped = cartesian_state_cur[:, np.newaxis, 6:8]
            distance_to_obstacles = np.linalg.norm(cartesian_state_reshaped - obstacles_reshaped, axis=2) # of shape (state_num, obstacle_num)
            collisions = (distance_to_obstacles < self.obstacles_radius[np.newaxis, :]).any(axis=1) # if a state's distance to any obstacle is less than radius, it collides
            return collisions
    def sdf(self, state_cur):
        """
        Compute the signed distance between the vehicle and the boundary of the track.

        This distance will be positive when the vehicle is inside the track and negative
        when it is outside.
        """
        if state_cur.ndim == 1:
            return min(self.track_width - state_cur[-2], self.track_width + state_cur[-2])
        else:
            sdf = np.minimum(self.track_width - state_cur[-2, :], self.track_width + state_cur[-2, :])
            return sdf


class MPPICollisionChecker(PointCollisionChecker):
    def __init__(self, obstacles=None, kinematics=None):
        CollisionChecker.__init__(self, obstacles, kinematics)
        self.check_other_agents = False
        self.other_agents_list = None

    def set_check_other_agents(self, check_other_agents):
        self.check_other_agents = check_other_agents

    def set_other_agents_list(self, other_agents_list):
        self.other_agents_list = other_agents_list

    def check(self, cartesian_state_cur, opponent_agents=None):  # True for collision, False for no collision
        #TODO: this function is called somewhere else, figure it out
        obstacle_collisions = self.check_collision_with_obstacles(cartesian_state_cur)
        agent_collisions = self.check_collisions_with_agents(cartesian_state_cur, opponent_agents=opponent_agents)
        # print(agent_collisions)
        return obstacle_collisions | agent_collisions

        # obstacle_collisions = self.check_collision_with_obstacles(cartesian_state_cur)
        # if self.check_other_agents:
        #     agent_collisions = self.check_collision_with_obstacles(cartesian_state_cur)
        #     return obstacle_collisions | agent_collisions
        # else:
        #     return obstacle_collisions

    def check_collisions_with_agents(self, cartesian_state_cur, opponent_agents=None):
        # import ipdb
        # ipdb.set_trace()
        # print(self.A)
        # print(self.other_agents_list)
        if opponent_agents is None:
            import ipdb
            ipdb.set_trace()
        assert opponent_agents is not None
        agents_as_obstacles = []
        agents_radius = []
        for agent in opponent_agents:
            agents_as_obstacles.append(agent.xy_coords)
            # jax.debug.print("x={x}", x = agents_as_obstacles)
            agents_radius.append(agent.radius)
        agents_as_obstacles, agents_radius = np.asarray(agents_as_obstacles), np.asarray(agents_radius)
        if cartesian_state_cur.ndim == 1:
            if (np.linalg.norm(agents_as_obstacles - cartesian_state_cur[:2], axis=1) < (agents_radius[np.newaxis, :] + self.kinematics.get_radius())).any():
                # if the agent collide with any other agents
                return True
            return False
        else:
            cartesian_state_cur = cartesian_state_cur.T
            obstacles_reshaped = agents_as_obstacles[np.newaxis, :, :]
            cartesian_state_reshaped = cartesian_state_cur[:, np.newaxis, :2]

            distance_to_obstacles = np.linalg.norm(cartesian_state_reshaped - obstacles_reshaped, axis=2) # of shape (state_num, obstacle_num)
            collisions = (distance_to_obstacles < (agents_radius[np.newaxis, :] + self.kinematics.get_radius()) ).any(axis=1) # if a state's distance to any obstacle is less than radius, it collides
            return collisions

    def check_collision_with_obstacles(self, cartesian_state_cur):
        if cartesian_state_cur.ndim == 1:
            if (np.linalg.norm(self.obstacles - cartesian_state_cur[:2], axis=1) < (self.obstacles_radius[np.newaxis, :] + self.kinematics.get_radius())).any():
                # if the vehicle collide with any one of the obstacles
                return True
            return False
        else:
            cartesian_state_cur = cartesian_state_cur.T
            obstacles_reshaped = self.obstacles[np.newaxis, :, :]
            cartesian_state_reshaped = cartesian_state_cur[:, np.newaxis, :2]

            distance_to_obstacles = np.linalg.norm(cartesian_state_reshaped - obstacles_reshaped, axis=2) # of shape (state_num, obstacle_num)
            collisions = (distance_to_obstacles < (self.obstacles_radius[np.newaxis, :] + self.kinematics.get_radius()) ).any(axis=1) # if a state's distance to any obstacle is less than radius, it collides
            return collisions

class Quadrotor2DCollisionChecker(CollisionChecker):
    def __init__(self, obstacles=None, kinematics=None):
        CollisionChecker.__init__(self, obstacles, kinematics)

    def initialize_from_config(self, config_data, section_name):
        CollisionChecker.initialize_from_config(self, config_data, section_name)
        self.obstacles = np.asarray(ast.literal_eval(config_data.get(section_name, "obstacles")))
        self.obstacles_radius = np.asarray(ast.literal_eval(config_data.get(section_name, "obstacles_radius")))
        kinematics_section_name = config_data.get(section_name, "kinematics")
        self.kinematics = factory_from_config(
            kinematics_factory_base, config_data, kinematics_section_name
        )

    def check(self, state_cur, opponent_agents=None, check_other_agents=None):  # True for collision, False for no collision
        obstacle_collisions = self.check_collision_with_obstacles(state_cur)
        agent_collisions = self.check_collisions_with_boundaries(state_cur)
        flip_collisions = self.check_collisions_with_drone_angle(state_cur)
        px_too_left = self.check_collision_with_px_left(state_cur)
        # print(agent_collisions)
        return obstacle_collisions | agent_collisions | flip_collisions | px_too_left
        # return self.check_collisions_with_boundaries(state_cur)

    def check_collision_with_px_left(self, state_cur):
        thresh = -4.5
        return state_cur[0] < thresh

    def check_collision_with_obstacles(self, cartesian_state_cur):
        if cartesian_state_cur.ndim == 1:
            if (np.linalg.norm(self.obstacles - cartesian_state_cur[:2], axis=1) < (self.obstacles_radius[np.newaxis, :] + self.kinematics.get_radius())).any():
                # if the vehicle collide with any one of the obstacles
                return True
            return False
        else:
            cartesian_state_cur = cartesian_state_cur.T
            obstacles_reshaped = self.obstacles[np.newaxis, :, :]
            cartesian_state_reshaped = cartesian_state_cur[:, np.newaxis, :2]

            distance_to_obstacles = np.linalg.norm(cartesian_state_reshaped - obstacles_reshaped, axis=2) # of shape (state_num, obstacle_num)
            collisions = (distance_to_obstacles < (self.obstacles_radius[np.newaxis, :] + self.kinematics.get_radius()) ).any(axis=1) # if a state's distance to any obstacle is less than radius, it collides
            return collisions

    def check_collisions_with_boundaries(self, state_cur):
        # ipdb.set_trace()
        is_oob = (state_cur[1] < 0)
        if state_cur.ndim == 1:
            # Single state. (6, )
            return is_oob
        else:
            # Trajectory. (6, T)
            collisions = np.where(is_oob, 1, 0)
            return collisions

    def check_collisions_with_drone_angle(self, state_cur):
        is_oob = (state_cur[2] < -np.pi/2) | (np.pi/2 < state_cur[2])
        if state_cur.ndim == 1:
            # Single state. (6, )
            return is_oob
        else:
            # Trajectory. (6, T)
            collisions = np.where(is_oob, 1, 0)
            return collisions


