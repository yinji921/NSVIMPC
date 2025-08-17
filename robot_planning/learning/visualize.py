import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd

from .dynamics import langevin_dynamics


def compare_ground_with_prediction(
    ground_truth, predictions, left_bound=-1.0, right_bound=1.0,
):
    plt.style.use('clean')
    plt.scatter(predictions[:, 0], predictions[:, 1], s=10)
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], s=10)

    plt.legend(['Predicted', 'True'])
    plt.axis('square')
    plt.xlim([left_bound, right_bound])
    plt.ylim([left_bound, right_bound])
    plt.grid()
    plt.show()

