import copy

import torch
import numpy as np
from robot_planning.environment.dynamics.autorally_dynamics import throttle_model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit
import os


FIT_INVERSE_THROTTLE = True

# dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path)
throttle_nn_file_path = "/home/ji/DCSL/robot_planning/environment/dynamics/autorally_dynamics/throttle_model1.pth"
throttle_nn = throttle_model.Net()
throttle_nn.load_state_dict(torch.load(throttle_nn_file_path))

if FIT_INVERSE_THROTTLE is False:
    _T = np.linspace(0, 2)  # for throttle model
else:
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

if FIT_INVERSE_THROTTLE is False:
    # For throttle model
    # A = np.array([T_f*0+1, T_f, wR_f, T_f**2, T_f**2*wR_f, T_f**2*wR_f**2, wR_f**2, T_f*wR_f**2, T_f*wR_f]).T
    # A = np.array([T_f*0+1, T_f, wR_f, T_f**2, wR_f**2, T_f*wR_f, T_f**3, wR_f**3, T_f**2*wR_f, T_f*wR_f**2, T_f**4, T_f**3*wR_f, T_f**2*wR_f**2, T_f*wR_f**3, wR_f**4]).T
    A = np.array(
        [
            T_f * 0 + 1,
            T_f,
            wR_f,
            T_f ** 2,
            wR_f ** 2,
            T_f * wR_f,
            T_f ** 3,
            wR_f ** 3,
            T_f ** 2 * wR_f,
            T_f * wR_f ** 2,
        ]
    ).T  # final choice
    B = result.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A, B)
    print(coeff)
    y = A @ coeff
    y = y.reshape(T.shape)
    # for throttle
    ax.plot_surface(
        T, wR, result, rstride=1, cstride=1, cmap="viridis", edgecolor="none"
    )
    ax.plot_wireframe(T, wR, y, color="black")

else:
    # For inverse throttle model
    # A = np.array([ac*0+1, ac, wR_f, ac**2, ac**2*wR_f, ac**2*wR_f**2, wR_f**2, ac*wR_f**2, ac*wR_f]).T
    # A = np.array([ac*0+1, ac, wR_f, ac**2, wR_f**2, ac*wR_f, ac**3, wR_f**3, ac**2*wR_f, ac*wR_f**2, ac**4, ac**3*wR_f, ac**2*wR_f**2, ac*wR_f**3, wR_f**4]).T
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
            ac ** 5,
        ]
    ).T  # final choice
    B = T.flatten()

    coeff, r, rank, s = np.linalg.lstsq(A, B)
    print(coeff)
    y = A @ coeff
    y = y.reshape(T.shape)
    # for inverse throttle
    ax.plot_surface(
        result, wR, T, rstride=1, cstride=1, cmap="viridis", edgecolor="none"
    )
    ax.plot_wireframe(result, wR, y, color="black")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
