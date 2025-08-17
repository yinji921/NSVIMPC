import numpy as np
import ast
import copy
from copy import deepcopy


class GoalChecker(object):
    def __init__(
        self,
        goal_state=None,
        goal_dim=None,
        goal_radius=None,
        goal_color=None,
        kinematics=None,
    ):
        self.kinematics = kinematics  # for more complicated collision checkers
        self.goal_state = goal_state
        self.goal_dim = goal_dim
        self.goal_radius = goal_radius
        self.goal_color = goal_color

    def initialize_from_config(self, config_data, section_name):
        pass

    def get_goal(self):
        return (self.goal_state[0], self.goal_state[1], self.goal_radius)

    def get_goal_color(self):
        return self.goal_color

    def set_goal(self, goal_state):
        self.goal_state = goal_state

    def check(self, state_cur):
        raise NotImplementedError

    def get_goal_dim(self):
        if self.goal_dim is not None:
            return self.goal_dim
        else:
            return len(self.goal_state)


class StateSpaceGoalChecker(GoalChecker):
    def __init__(self, goal=None, kinematics=None):
        GoalChecker.__init__(self, goal, kinematics)

    def initialize_from_config(self, config_data, section_name):
        GoalChecker.initialize_from_config(self, config_data, section_name)
        self.goal_state = np.asarray(
            ast.literal_eval(config_data.get(section_name, "goal_state"))
        ).reshape((-1, 1))
        self.goal_radius = config_data.getfloat(section_name, "goal_radius")
        if config_data.has_option(section_name, "goal_color"):
            self.goal_color = config_data.get(section_name, "goal_color")
        else:
            self.goal_color = "r"

    def check(self, state_cur):  # True for goal reached, False for goal not reached
        if np.linalg.norm(self.goal_state - state_cur) < self.goal_radius:
            return True
        return False


class PositionGoalChecker(GoalChecker):
    def __init__(self, goal=None, kinematics=None):
        GoalChecker.__init__(self, goal, kinematics)

    def initialize_from_config(self, config_data, section_name):
        GoalChecker.initialize_from_config(self, config_data, section_name)
        self.goal_state = np.asarray(
            ast.literal_eval(config_data.get(section_name, "goal_state"))
        ).reshape((-1, 1))
        self.goal_radius = config_data.getfloat(section_name, "goal_radius")
        if config_data.has_option(section_name, "goal_color"):
            self.goal_color = str(config_data.get(section_name, "goal_color"))
        else:
            self.goal_color = "r"

    def check(self, state_cur):  # True for goal reached, False for goal not reached
        pos_cur = np.array([state_cur[0], state_cur[1]])
        goal_pos = np.array([self.goal_state[0, 0], self.goal_state[1, 0]])
        if np.linalg.norm(goal_pos - pos_cur) < self.goal_radius:
            return True
        return False

    def dist(self, state_cur):
        pos_cur = np.array([state_cur[0], state_cur[1]])
        return np.linalg.norm(self.goal_state - pos_cur)


class FlexStateSpaceGoalChecker(GoalChecker):
    def __init__(self, goal=None, kinematics=None):
        GoalChecker.__init__(self, goal, kinematics)

    def initialize_from_config(self, config_data, section_name):
        GoalChecker.initialize_from_config(self, config_data, section_name)
        self.goal_state = None
        self.goal_dim = (int(config_data.getfloat(section_name, "goal_dim")),)
        self.goal_radius = config_data.getfloat(section_name, "goal_radius")
        if config_data.has_option(section_name, "goal_color"):
            self.goal_color = config_data.get(section_name, "goal_color")
        else:
            self.goal_color = "r"

    def get_goal(self):
        return (self.goal_state[0], self.goal_state[1], self.goal_radius)

    def set_goal(self, goal_state):
        self.goal_state = goal_state

    def check(self, state_cur):  # True for goal reached, False for goal not reached
        if (
            np.linalg.norm(np.squeeze(self.goal_state) - np.squeeze(state_cur))
            < self.goal_radius
        ):
            return True
        return False


class AutorallyCartesianGoalChecker(GoalChecker):
    def __init__(self, obstacles=None, kinematics=None):
        GoalChecker.__init__(self, obstacles, kinematics)

    def initialize_from_config(self, config_data, section_name):
        GoalChecker.initialize_from_config(self, config_data, section_name)
        self.goal_state = np.asarray(
            ast.literal_eval(config_data.get(section_name, "goal_state"))
        )
        self.goal_radius = config_data.getfloat(section_name, "goal_radius")
        if config_data.has_option(section_name, "goal_color"):
            self.goal_color = config_data.get(section_name, "goal_color")
        else:
            self.goal_color = "r"

    def get_goal(self):
        return (self.goal_state[-2], self.goal_state[-1], self.goal_radius)

    def check(self, state_cur):
        state_cur = np.squeeze(state_cur)
        if np.linalg.norm(self.goal_state[-2:] - state_cur[-2:]) < self.goal_radius:
            return True
        return False


class QuadrotorCartesianGoalChecker(GoalChecker):
    def __init__(self, obstacles=None, kinematics=None):
        GoalChecker.__init__(self, obstacles, kinematics)

    def initialize_from_config(self, config_data, section_name):
        GoalChecker.initialize_from_config(self, config_data, section_name)

        # goal_state is just the x-position of the goal. px >= goal_state.
        self.goal_state = np.asarray(
            ast.literal_eval(config_data.get(section_name, "goal_state"))
        )
        # self.goal_radius = config_data.getfloat(section_name, "goal_radius")
        if config_data.has_option(section_name, "goal_color"):
            self.goal_color = config_data.get(section_name, "goal_color")
        else:
            self.goal_color = "r"

    def get_goal(self):
        return self.goal_state

    def check(self, state_cur):
        state_cur = np.squeeze(state_cur)
        px = state_cur[0]
        reached_goal = px >= self.goal_state[0]
        return reached_goal
        # if np.linalg.norm(self.goal_state[:3] - state_cur[:3]) < self.goal_radius:
        #     return True
        # return False

class DubinsCartesianGoalChecker(GoalChecker):
    def __init__(self, obstacles=None, kinematics=None):
        GoalChecker.__init__(self, obstacles, kinematics)

    def initialize_from_config(self, config_data, section_name):
        GoalChecker.initialize_from_config(self, config_data, section_name)

        # goal_state is just the x-position of the goal. px >= goal_state.
        self.goal_state = np.asarray(
            ast.literal_eval(config_data.get(section_name, "goal_state"))
        )
        self.goal_radius = config_data.getfloat(section_name, "goal_radius")
        # self.goal_radius = config_data.getfloat(section_name, "goal_radius")
        if config_data.has_option(section_name, "goal_color"):
            self.goal_color = config_data.get(section_name, "goal_color")
        else:
            self.goal_color = "r"

    def get_goal(self):
        return self.goal_state

    def check(self, state_cur):
        state_cur = np.squeeze(state_cur)
        if np.linalg.norm(self.goal_state[:2] - state_cur[:2]) < self.goal_radius:
            return True
        return False


