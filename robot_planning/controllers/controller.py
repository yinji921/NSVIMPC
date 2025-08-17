from robot_planning.environment.dynamics.simulated_dynamics import SimulatedDynamics
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factories import cost_evaluator_factory_base
import copy


class Controller(object):
    def __init__(
        self,
        dynamics: SimulatedDynamics = None,
        cost_evaluator=None,
        control_dim=None,
        renderer=None,
        warm_start_itr=1,
    ):
        self.dynamics = dynamics
        self.cost_evaluator = cost_evaluator
        self.control_dim = control_dim
        self.renderer = renderer
        self.warm_start_itr = warm_start_itr

    def initialize_from_config(self, config_data, section_name):
        dynamics_section_name = config_data.get(section_name, "dynamics")
        self.dynamics = factory_from_config(
            dynamics_factory_base, config_data, dynamics_section_name
        )
        cost_evaluator_section_name = config_data.get(section_name, "cost_evaluator")
        self.cost_evaluator = factory_from_config(
            cost_evaluator_factory_base, config_data, cost_evaluator_section_name
        )
        self.control_dim = config_data.getint(section_name, "control_dim")
        if config_data.has_option(section_name, "warm_start_itr"):
            self.warm_start_itr = config_data.getint(section_name, "warm_start_itr")
        else:
            self.warm_start_itr = 1
        if self.control_dim != self.dynamics.get_action_dim()[0]:
            raise ValueError(
                "The Controller 's control dimension and dynamics' control dimension do not match! "
            )

    def plan(self, state_cur, warm_start=False):
        raise NotImplementedError

    def get_control_dim(self):
        return copy.copy(self.control_dim)

    def get_dynamics(self):
        return copy.deepcopy(self.dynamics)

    def get_cost_evaluator(self):
        return copy.deepcopy(self.cost_evaluator)

    def set_control_dim(self, control_dim):
        self.control_dim = control_dim

    def set_dynamics(self, dynamics):
        self.dynamics = dynamics

    def set_cost_evaluator(self, cost_evaluator):
        self.cost_evaluator = cost_evaluator

    def set_renderer(self, renderer):
        self.renderer = renderer

    def reset(self):
        raise NotImplementedError


class MpcController(Controller):
    def __init__(
        self,
        control_horizon=None,
        dynamics=None,
        cost_evaluator=None,
        control_dim=None,
        renderer=None,
    ):
        Controller.__init__(self, dynamics, cost_evaluator, control_dim, renderer)
        self.control_horizon = control_horizon

    def initialize_from_config(self, config_data, section_name):
        Controller.initialize_from_config(self, config_data, section_name)
        self.control_horizon = config_data.getint(section_name, "control_horizon")

    def plan(self, state_cur, warm_start=False):
        raise NotImplementedError

    def get_control_horizon(self):
        return copy.copy(self.control_horizon)

    def set_control_horizon(self, control_horizon):
        self.control_horizon = control_horizon
