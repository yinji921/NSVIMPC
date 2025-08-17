from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factories import cost_evaluator_factory_base
from robot_planning.factory.factories import (
    dynamic_action_bounds_generator_factory_base,
)
from robot_planning.factory.factories import controller_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factories import observer_factory_base
from robot_planning.factory.factories import robot_factory_base
from robot_planning.environment.robots.base_robot import Robot
import numpy as np
import copy
from copy import deepcopy
import ast


class AbstractRobot(Robot):
    def __init__(
        self,
        dynamics=None,
        simulated_robot=None,
        start_state=None,
        steps_per_action=None,
        abstract_action_horizon=None,
        data_type=None,
        cost_evaluator=None,
        controller=None,
        renderer=None,
        observer=None,
    ):
        Robot.__init__(self)
        self.dynamics = dynamics
        self.start_state = start_state
        self.cost_evaluator = cost_evaluator
        self.data_type = data_type
        self.steps_per_action = steps_per_action
        self.steps = 0
        self.controller = controller
        self.renderer = renderer
        self.observer = observer
        self.abstract_action_horizon = abstract_action_horizon
        self.trajectory_cost = 0

        self.is_dynamic_action_bounds = False
        self.dynamic_action_bounds_generator = None

    def initialize_from_config(self, config_data, section_name):
        Robot.initialize_from_config(self, config_data, section_name)
        dynamics_section_name = config_data.get(section_name, "dynamics")
        self.dynamics = factory_from_config(
            dynamics_factory_base, config_data, dynamics_section_name
        )
        self.abstract_action_horizon = int(
            config_data.get(section_name, "abstract_action_horizon")
        )
        self.start_state = np.asarray(
            ast.literal_eval(config_data.get(section_name, "start_state"))
        )
        self.data_type = config_data.get(section_name, "data_type")
        if config_data.has_option(section_name, "steps_per_action"):
            self.steps_per_action = config_data.getint(section_name, "steps_per_action")
        else:
            self.steps_per_action = 1

        # controller may have a different cost evaluator from the robot, if we use the robot to train rl algorithms
        if config_data.has_option(section_name, "cost_evaluator"):
            cost_evaluator_section_name = config_data.get(
                section_name, "cost_evaluator"
            )
            self.cost_evaluator = factory_from_config(
                cost_evaluator_factory_base, config_data, cost_evaluator_section_name
            )
            # link the goal checker of the simulated robot with the abstract robot's cost evaluator sub_goal_checker
            self.cost_evaluator.set_sub_goal_checker(
                self.dynamics.simulated_robot.cost_evaluator.goal_checker
            )
        if config_data.has_option(section_name, "observer"):
            observer_section_name = config_data.get(section_name, "observer")
            self.observer = factory_from_config(
                observer_factory_base, config_data, observer_section_name
            )

        self.dynamics.simulated_robot.state = self.start_state
        self.cost_evaluator.set_collision_checker(
            self.dynamics.simulated_robot.cost_evaluator.collision_checker
        )

        if config_data.has_option(section_name, "dynamic_action_bounds_generator"):
            self.is_dynamic_action_bounds = True
            dab_section_name = config_data.get(
                section_name, "dynamic_action_bounds_generator"
            )
            self.dynamic_action_bounds_generator = factory_from_config(
                dynamic_action_bounds_generator_factory_base,
                config_data,
                dab_section_name,
            )
            self.dynamic_action_bounds_generator.set_max_range(
                self.dynamics.simulated_robot.dynamics.state_bounds
            )

    @property
    def delta_t(self):
        return self.dynamics.get_delta_t()

    @property
    def state(self):
        return copy.copy(self.dynamics.simulated_robot.state)

    @state.setter
    def state(self, x):
        assert x.shape == self.get_state_dim()
        self.dynamics.simulated_robot.set_state(x)

    def get_state(self):
        return copy.copy(self.dynamics.simulated_robot.state)

    def get_time(self):
        return self.steps * self.abstract_action_horizon * self.dynamics.get_delta_t()

    def get_state_dim(self):
        return self.dynamics.get_state_dim()

    def get_action_dim(self):
        return (
            self.dynamics.simulated_robot.cost_evaluator.goal_checker.goal_state.shape
        )  # action is the goal in the state space

    def get_obs_dim(self):
        return self.observer.get_obs_dim()

    def get_model_base_type(self):
        return self.dynamics.base_type

    def get_data_type(self):
        return self.data_type

    def get_renderer(self):
        return self.renderer

    def set_state(self, x):
        assert x.shape == self.get_state_dim()
        self.dynamics.simulated_robot.set_state(x)

    def set_time(self, time):
        self.steps = time / self.dynamics.get_delta_t()

    def reset_time(self):
        self.steps = 0

    def reset_state(self, initial_state, random):
        if random:
            state_shape = self.dynamics.get_state_dim()
            state_bounds = self.dynamics.get_state_bounds()

            new_state = []
            for i in range(state_shape[0]):
                new_state.append(
                    np.random.random() * 2 * state_bounds[i] - state_bounds[i]
                )
            new_state = np.array(new_state)
            self.state = deepcopy(new_state)

        else:
            if initial_state is not None:
                self.state = initial_state
            else:
                self.state = deepcopy(self.start_state)

    def reset_controller(self):
        self.dynamics.simulated_robot.reset_controller()

    def set_cost_evaluator(self, cost_evaluator):
        self.cost_evaluator = cost_evaluator

    def set_renderer(self, renderer):
        # the renderer is only responsible for renderering the robot itself and visualizing its controller info(such as MPPI)
        self.renderer = renderer
        self.dynamics.simulated_robot.set_renderer(renderer)

    def render_robot_state(self):
        self.dynamics.simulated_robot.render_robot_state()
        # if self.dynamics.simulated_robot.renderer is not None:
        #     self.dynamics.simulated_robot.render_robot_state()
        # self.dynamics.simulated_robot.renderer.render_states(state_list=[self.get_state()],
        #                                                      kinematics=self.dynamics.simulated_robot.controller.cost_evaluator.collision_checker.kinematics)

    def render_obstacles(self):
        if self.renderer is not None:
            obstacle_list = self.cost_evaluator.collision_checker.get_obstacle_list()
            self.renderer.render_obstacles(
                obstacle_list=obstacle_list, **{"color": "k"}
            )

    def render_goal(self):
        if self.renderer is not None:
            goal = self.cost_evaluator.goal_checker.get_goal()
            goal_color = self.cost_evaluator.goal_checker.get_goal_color()
            self.renderer.render_goal(goal=goal, **{"color": goal_color, "alpha": 0.8})

    def prepare_to_take_action(self, action):
        assert isinstance(
            action, np.ndarray
        ), "simulated robot has numpy.ndarray type action!"
        # reset trajectory cost
        self.trajectory_cost = 0

        # update goal for the controller
        action = action.reshape(self.get_action_dim())
        self.dynamics.simulated_robot.controller.cost_evaluator.goal_checker.set_goal(
            action
        )
        self.dynamics.simulated_robot.cost_evaluator.goal_checker.set_goal(action)

    def single_step_simulated_robot(self):
        self.render_goal()
        state_next, cost_t = self.dynamics.simulated_robot.take_action_with_controller()
        if self.cost_evaluator.goal_checker.check(state_next):
            cost_t += self.cost_evaluator.ultimate_goal_cost

        self.trajectory_cost += cost_t

    def collect_trajectory_info(self):
        state_next = self.get_state()
        cost = self.trajectory_cost
        return state_next, cost

    def propagate_robot(self, action):
        assert isinstance(
            action, np.ndarray
        ), "simulated robot has numpy.ndarray type action!"

        # update goal for the controller
        action = action.reshape(self.get_action_dim())
        assert all(action <= self.generate_dynamic_action_bounds()[1]) and all(
            action >= self.generate_dynamic_action_bounds()[0]
        )

        self.dynamics.simulated_robot.controller.cost_evaluator.goal_checker.set_goal(
            action
        )
        self.dynamics.simulated_robot.cost_evaluator.goal_checker.set_goal(action)
        state_next, cost = None, 0
        for _ in range(self.abstract_action_horizon):
            (
                state_next,
                cost_t,
            ) = self.dynamics.simulated_robot.take_action_with_controller()
            self.render_goal()
            cost += cost_t
        self.steps += 1
        return state_next, cost

    def evaluate_state_action_pair_cost(self, state, action, state_next):
        return self.cost_evaluator.evaluate(
            state_cur=state, action=action, state_next=state_next
        )

    def take_action(self, action):
        assert isinstance(
            action, np.ndarray
        ), "simulated robot has numpy.ndarray type action!"
        state_cur = copy.deepcopy(self.state)
        state_next = None
        cost = 0
        self.render_robot_state()
        for _ in range(self.steps_per_action):
            state_next, cost_step = self.propagate_robot(action)
            cost += cost_step
        assert state_next is not None, "invalid state!"

        # check if assigned new goal has been achieved
        self.evaluate_state_action_pair_cost(
            state=state_cur, action=action, state_next=state_next
        )
        return state_next, cost

    def take_action_sequence(self, actions):
        assert isinstance(
            actions, np.ndarray
        ), "simulated robot has numpy.ndarray type action!"
        state_next = None
        cost = 0
        self.render_robot_state()
        for action in actions:
            state_next, cost_step = self.propagate_robot(action)
            cost += cost_step
        assert state_next is None, "invalid state!"
        return state_next, cost

    def generate_dynamic_action_bounds(self):
        if self.dynamic_action_bounds_generator is None:
            return None
        else:
            action_bounds = self.dynamic_action_bounds_generator.generate_action_bounds(
                copy.copy(self.state)
            )
            return action_bounds
