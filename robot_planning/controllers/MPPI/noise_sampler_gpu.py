import numpy as np
import ast
from robot_planning.controllers.MPPI.noise_sampler import NoiseSampler


class GaussianNoiseSamplerGpu(NoiseSampler):
    def __init__(self, mean=None, covariance=None):
        NoiseSampler.__init__(self)
        self.mean = mean
        self.covariance = covariance

    def initialize_from_config(self, config_data, section_name):
        self.mean = np.asarray(ast.literal_eval(config_data.get(section_name, "mean")))
        self.covariance = np.asarray(
            ast.literal_eval(config_data.get(section_name, "covariance"))
        )

    """
    def sample(self, control_dim, control_horizon):
        assert (control_dim == self.control_dim)
        assert (control_horizon == self.control_horizon)

        if self.mean.shape[0] is not control_dim or self.covariance.shape[0] is not control_dim:
            raise ValueError('noise dimensions and control dimensions do not match!')
        noises = np.random.multivariate_normal(self.mean, self.covariance, size=control_horizon).reshape((control_dim, control_horizon))
        # if not(noises.shape[0] is control_dim and noises.shape[1] is control_horizon):
        #     raise ValueError('noise dimensionality {} is not the expected value ({}, {})'.format(noises.shape, control_dim, control_horizon))
        return noises
    """
