import torch
import numpy as np
from robot_planning.environment.dynamics.autorally_dynamics import throttle_model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit

throttle_nn_file_path = "environment/dynamics/autorally_dynamics/throttle_model1.pth"
throttle_nn = throttle_model.Net()
throttle_nn.load_state_dict(torch.load(throttle_nn_file_path))

_T = np.linspace(0, 10)
_wR = np.linspace(20, 100)
T, wR = np.meshgrid(_T, _wR)

throttle_factor = 0.45
input_tensor = torch.from_numpy(
    np.hstack((T.reshape((-1, 1)), wR.reshape((-1, 1)) / throttle_factor))
).float()
result = throttle_nn(input_tensor).detach().numpy().flatten()
result = result.reshape(T.shape)

# fit function
xdata = np.vstack([T.flatten(), wR.flatten()])
ydata = result.flatten()


def func():
    L1W = np.array(
        [
            [-4.0160503387e01, -1.0396156460e-01],
            [5.6865772247e01, 1.1424105167e00],
            [-2.7877355576e01, 8.1013500690e-02],
            [-2.7034116745e01, -2.0072770119e00],
            [3.0620580673e01, -1.6317946836e-02],
            [-3.4583007812e01, 2.2306546569e-02],
            [1.7588891983e01, -1.1739752442e-01],
            [-4.2060924530e01, 4.9316668883e-03],
            [2.8054933548e00, 6.7983172834e-02],
            [-2.0927755356e01, -2.7351467609e00],
            [-3.4222785950e01, -8.8916033506e-02],
            [-5.0679687500e01, -4.3296713829e00],
            [-3.8273872375e01, 6.3557848334e-02],
            [-6.8535380363e00, 1.0541532934e-01],
            [-5.7073421478e01, -1.8881739378e00],
            [3.9726573944e01, 5.3338561207e-02],
            [-3.1062200546e01, -3.2722020149e00],
            [3.2069881439e01, 2.1122207642e02],
            [-1.4632539368e02, 5.0957229614e01],
            [-7.5698246956e00, -1.2893879414e-01],
        ]
    )
    L1B = np.array(
        [
            4.9647192955,
            -11.6490354538,
            0.9536626935,
            30.6506881714,
            -7.0774221420,
            0.9992623329,
            10.6967773438,
            9.4048290253,
            -17.5587844849,
            1.5835133791,
            3.5548210144,
            6.9317874908,
            5.8543529510,
            -17.1659870148,
            7.7465252876,
            -9.3858194351,
            34.2410469055,
            -7.3937258720,
            -204.8706054688,
            0.6013393998,
        ]
    )
    L2W = np.array(
        [
            [
                6.1452589035,
                1.0189874172,
                -25.4556179047,
                3.2313644886,
                43.4599952698,
                20.1596012115,
                17.1203384399,
                -22.8726711273,
                -90.1767959595,
                3.3784775734,
                5.1697301865,
                8.3276357651,
                -28.8062953949,
                -26.1210975647,
                9.5909786224,
                28.0447864532,
                2.8735139370,
                3.8586735725,
                3.5681331158,
                -8.9000215530,
            ]
        ]
    )
    L2B = np.array(5.8514204025)
    inputs = np.hstack((_T.reshape((-1, 1)), _wR.reshape((-1, 1)) / throttle_factor))
    val = np.matmul(1 / (1 + np.exp(-(np.matmul(inputs, L1W.T) + L1B))), L2W.T) + L2B
    return val


_wR = wR.flatten()
_T = T.flatten()
fit_result = func()
fit_result = fit_result.reshape(T.shape)
print(result - fit_result)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(T, wR, result, rstride=1, cstride=1, cmap="viridis", edgecolor="none")
ax.plot_wireframe(T, wR, fit_result, color="black")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
