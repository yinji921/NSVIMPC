import copy

import torch
import numpy as np
from robot_planning.environment.dynamics.autorally_dynamics import throttle_model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit
import os


def fitted_inverse_throttle_model(ac, wR_f):
    A = np.array(
        [
            ac * 0 + 1,
            ac,
            wR_f,
            ac ** 2,
            wR_f ** 2,
            ac * wR_f,
            ac ** 3,
            wR_f ** 3,
            ac ** 2 * wR_f,
            ac * wR_f ** 2,
        ]
    ).T
    coeff = np.array(
        [
            -1.64097548e-01,
            2.59430623e-03,
            2.13849602e-02,
            -3.20810390e-05,
            -4.23167078e-04,
            -7.85209523e-05,
            6.87524700e-07,
            2.88939897e-06,
            -6.36897315e-08,
            9.76213705e-07,
        ]
    )
    return A @ coeff


if __name__ == "__main__":
    throttle_nn_file_path = (
        "environment/dynamics/autorally_dynamics/throttle_model1.pth"
    )
    throttle_nn = throttle_model.Net()
    throttle_nn.load_state_dict(torch.load(throttle_nn_file_path))
    _T = np.linspace(0, 1.0)  # for inverse throttle model

    _wR = np.linspace(20, 100)
    T, wR = np.meshgrid(_T, _wR)
    throttle_factor = 0.45
    input_tensor = torch.from_numpy(
        np.hstack((T.reshape((-1, 1)), wR.reshape((-1, 1)) / throttle_factor))
    ).float()
    result = throttle_nn(input_tensor).detach().numpy().flatten()
    result = result.reshape(T.shape)

    T_f = T.flatten()
    wR_f = wR.flatten()
    ac = result.flatten()

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    y = fitted_inverse_throttle_model(ac, wR_f)
    y = y.reshape(T.shape)
    # for inverse throttle
    ax.plot_surface(
        result, wR, T, rstride=1, cstride=1, cmap="viridis", edgecolor="none"
    )
    ax.plot_wireframe(result, wR, y, color="black")

    ax.set_xlabel("rear_wheel_acceleration")
    ax.set_ylabel("rear_wheel_speed")
    ax.set_zlabel("throttle")
    plt.show()
