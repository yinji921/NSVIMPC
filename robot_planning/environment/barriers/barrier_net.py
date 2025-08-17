from math import pi
from collections import OrderedDict
from robot_planning.factory.factories import collision_checker_factory_base
from robot_planning.factory.factory_from_config import factory_from_config
import jax.numpy as np
import jax
import os

import torch
import torch.nn as nn


class BarrierNN(torch.nn.Module):
    def __init__(
        self,
        hidden_layers: int = 2,
        hidden_layer_width: int = 16,
        n_inputs: int = 9,
    ):
        """
        A model for learning a barrier function.

        We'll learn the barrier function as the output of a neural network

        args:
            hidden_layers: how many hidden layers to have
            hidden_layer_width: how many neurons per hidden layer
            n_inputs: how many input state dimensions
        """
        super(BarrierNN, self).__init__()

        # Construct the network
        self.layers = OrderedDict()
        self.layers["input_linear"] = nn.Linear(
            n_inputs,
            hidden_layer_width,
        )
        for i in range(hidden_layers):
            self.layers[f"layer_{i}_activation"] = nn.Tanh()
            self.layers[f"layer_{i}_linear"] = nn.Linear(
                hidden_layer_width, hidden_layer_width
            )
        self.layers["output_activation"] = nn.Tanh()
        self.layers["output_linear"] = nn.Linear(hidden_layer_width, 1)
        self.nn = nn.Sequential(self.layers)

        # Load weights
        dir_path = os.path.dirname(os.path.realpath(__file__))
        saved_data = torch.load(os.path.join(dir_path, "autorally_barrier_net.pth"), weights_only=True)
        self.load_state_dict(saved_data["model"])
        self.alpha = saved_data["alpha"]

        self.n_inputs = n_inputs
        self.state_scale = [20, 20, 2 * pi, 100, 100, 0.7 * pi, 3.0 / 10, 20.0]

        self.obstacles = False
        self.obstacles_cbf_coefficient = 3.0

    def initialize_from_config(self, config_data, section_name):
        collision_checker_section_name = config_data.get(section_name, "collision_checker")
        self.collision_checker = factory_from_config(
            collision_checker_factory_base,
            config_data,
            collision_checker_section_name
        )
        if config_data.has_option(section_name, "obstacles"):
            self.obstacles = config_data.getboolean(section_name, "obstacles")
        if config_data.has_option(section_name, "obstacles_cbf_coefficient"):
            self.obstacles_cbf_coefficient = config_data.getfloat(section_name, "obstacles_cbf_coefficient")

    def nn_input(self, x):
        x = torch.from_numpy(x)
        nn_input = torch.zeros(x.shape[0], self.n_inputs)
        nn_input[:, :-2] = x[:, :-1]  # ignore s
        nn_input[:, -2] = torch.sin(x[:, -3])  # add sin e_psi
        nn_input[:, -1] = torch.cos(x[:, -3])  # add cos e_psi

        # scale by state bounds
        scale = torch.ones_like(nn_input)
        scale[:, :-2] = torch.tensor([1 / scale for scale in self.state_scale[:-1]])
        nn_input = nn_input * scale

        return nn_input

    def danger_index(self, x, x_global=None):
        """Negative in the safe region, positive in the unsafe region."""
        track_width = self.collision_checker.track_width
        # track_width = 1.5 # overwrite trackwidth
        danger = x[:, -2] ** 2 - track_width**2  # don't run off the side
        if (x_global is not None) and (self.obstacles):
            ###########################
            x_global = x_global.T
            obstacles_reshaped = self.collision_checker.obstacles[np.newaxis, :, :]
            cartesian_state_reshaped = x_global[:, np.newaxis, 6:8]
            distance_to_obstacles = np.linalg.norm(cartesian_state_reshaped - obstacles_reshaped, axis=2) # of shape (state_num, obstacle_num)
            obstacles_danger = self.obstacles_cbf_coefficient * (self.collision_checker.obstacles_radius[np.newaxis, :]**2 - distance_to_obstacles**2)
            obstacles_danger = np.max(obstacles_danger, axis=1)
            # jax.debug.print("{x}", x=obstacles_danger)
            danger = np.stack((danger, obstacles_danger), axis=1)
            danger = np.max(danger, axis=1)
            ###########################
        return danger

    def forward(self, x, x_global=None):
        # h = self.nn(self.nn_input(x))  # batch size x 1
        # return self.danger_index(x).squeeze() + 0 * h.squeeze().detach().cpu().numpy()
        # Ignore the nn
        return self.danger_index(x, x_global).squeeze()
