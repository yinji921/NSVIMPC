import copy

import torch
import numpy as np
from robot_planning.environment.dynamics.autorally_dynamics import throttle_model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit
import os


def fitted_inverse_throttle_model(ac, wR_f):
    ac = np.asarray(ac, dtype=np.float64)
    ac[ac > 99.9999999999] = 99.9999999999
    throttle = 0.2000 + (-np.log((-ac + 100) / 0.00625) + 10.00000) / 33.33333
    return throttle


if __name__ == "__main__":
    throttle_nn_file_path = (
        "environment/dynamics/autorally_dynamics/throttle_model1.pth"
    )
    throttle_nn = throttle_model.Net()
    throttle_nn.load_state_dict(torch.load(throttle_nn_file_path))
    _T = np.linspace(0, 1.0, 100)  # for inverse throttle model

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
