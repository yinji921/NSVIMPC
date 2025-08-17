from robot_planning.controllers.MPPI.stochastic_trajectories_sampler import MPPIStochasticTrajectoriesSampler
import einops as ei
import numpy as onp
from robot_planning.controllers.controller import Controller
# from robot_planning.environment.renderers import AutorallyMatplotlibRenderer
from robot_planning.environment.mpl_renderer import AutorallyMatplotlibRenderer
from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import cost_evaluator_factory_base
from robot_planning.factory.factories import (
    stochastic_trajectories_sampler_factory_base,
)
import time
from robot_planning.environment.barriers.barrier_net import BarrierNN
import jax
import jax.numpy as np
from jax.scipy.optimize import minimize
from functools import partial
import ast
import copy
import scipy.optimize as sciopt
from robot_planning.helper.timer import Timer


class Quadrotor2DOpenLoop(Controller):
    def __init__(self):
        super(Quadrotor2DOpenLoop, self).__init__()

    def initialize_from_config(self, config_data, section_name):
        Controller.initialize_from_config(self, config_data, section_name)

    def plan(self, state_cur, warm_start=None, opponent_agents=None, logger=None):
        u = np.asarray([0.051, 0.05])
        return u
