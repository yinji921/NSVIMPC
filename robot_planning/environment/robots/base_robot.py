from robot_planning.factory.factory_from_config import factory_from_config
from robot_planning.factory.factories import dynamics_factory_base
from robot_planning.factory.factories import cost_evaluator_factory_base
from robot_planning.factory.factories import controller_factory_base
from robot_planning.factory.factories import renderer_factory_base
from robot_planning.factory.factories import observer_factory_base
import numpy as np
import copy
from copy import deepcopy
import ast


class Robot(object):
    def __init__(self):
        pass

    def initialize(self):
        pass

    def initialize_from_config(self, config_data, section_name):
        pass

    def initialize_loggers(self):
        pass

    def get_state(self):
        raise NotImplementedError

    def take_action(self, action):
        raise NotImplementedError

    def take_action_sequence(self, actions):
        raise NotImplementedError
