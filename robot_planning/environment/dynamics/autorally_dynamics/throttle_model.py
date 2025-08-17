import torch
from torch.nn import functional as F
import numpy as np
import scipy.io
import itertools
import matplotlib.pyplot as plt

from flax import linen as nn

# import pandas
from numpy import (
    sin,
    cos,
    tan,
    arctan as atan,
    sqrt,
    arctan2 as atan2,
    zeros,
    zeros_like,
    abs,
    pi,
)
import time


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cuda = torch.device("cpu")

        self.linear1 = torch.nn.Linear(2, 20)
        self.linear2 = torch.nn.Linear(20, 1)

    def forward(self, inputs):
        output = self.linear2(torch.sigmoid(self.linear1(inputs)))
        return output


class JaxNet:
    def __init__(self, torch_model):
        self.linear1 = nn.Dense(features=20)
        self.linear2 = nn.Dense(features=1)

        # Get layer 1 params
        # (see https://flax.readthedocs.io/en/latest/advanced_topics/convert_pytorch_to_flax.html)
        kernel = torch_model.linear1.weight.detach().cpu().numpy()
        kernel = np.transpose(kernel, (1, 0))
        bias = torch_model.linear1.bias.detach().cpu().numpy()
        self.l1_params = {'params': {'kernel': kernel, 'bias': bias}}

        # Get layer 2 params
        kernel = torch_model.linear2.weight.detach().cpu().numpy()
        kernel = np.transpose(kernel, (1, 0))
        bias = torch_model.linear2.bias.detach().cpu().numpy()
        self.l2_params = {'params': {'kernel': kernel, 'bias': bias}}

    def __call__(self, inputs):
        # Compute output
        output = self.linear1.apply(self.l1_params, inputs)
        output = nn.sigmoid(output)
        output = self.linear2.apply(self.l2_params, output)
        return output


def prepare_data():
    N0 = 0
    Nf = -1
    mat = scipy.io.loadmat("throttle_id/ramp.mat")
    throttle1 = mat["throttle"][N0::25].T
    dx, N1 = throttle1.shape
    wR1 = mat["wR"][N0::25].T
    du, N1 = wR1.shape

    mat = scipy.io.loadmat("throttle_id/impulse.mat")
    throttle2 = mat["throttle"][N0::25].T
    dx, N2 = throttle2.shape
    wR2 = mat["wR"][N0::25].T
    du, N2 = wR2.shape

    N0 = 400
    Nf = -200
    mat = scipy.io.loadmat("mppi_data/mppi_states_controls1.mat")
    states = mat["states"][:, ::10]
    controls = mat["inputs"][:, ::10]
    wR1 = states[4:5, N0 : Nf - 1 : 25]
    throttle1 = controls[1:, N0 : Nf - 1 : 25]

    N0 = 400
    Nf = -200
    mat = scipy.io.loadmat("mppi_data/mppi_states_controls2.mat")
    states = mat["states"][:, ::10]
    controls = mat["inputs"][:, ::10]
    wR2 = states[4:5, N0 : Nf - 1 : 25]
    throttle2 = controls[1:, N0 : Nf - 1 : 25]

    training_inputs = np.hstack(
        (
            np.vstack((throttle1[0, :-1], wR1[0, :-1])),
            np.vstack((throttle2[0, :-1], wR2[0, :-1])),
        )
    )
    training_outputs = np.hstack((wR1[0, 1:], wR2[0, 1:])).reshape((1, -1))

    return training_inputs, training_outputs


def train_model():
    dyn_model = Net()
    # dyn_model.load_state_dict(torch.load('throttle_model1.pth'))
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(dyn_model.parameters(), lr=1e-2)
    training_inputs, training_outputs = prepare_data()
    input_tensor = torch.from_numpy(training_inputs).float().T
    output_tensor = torch.from_numpy(training_outputs).float().T
    for ii in range(50000):
        optimizer.zero_grad()
        dwR = dyn_model(input_tensor)
        wR = input_tensor[:, 1:] + dwR * 0.25
        # print(output.shape, output_tensor.shape)
        loss = criterion(wR, output_tensor)
        loss.backward()
        optimizer.step()
        if ii % 100 == 0:
            print(ii, loss.item())
    torch.save(dyn_model.state_dict(), "throttle_model2.pth")

    dyn_model.load_state_dict(torch.load("throttle_model2.pth"))
    dwR = dyn_model(input_tensor)
    wR = dwR * 0.25 + input_tensor[:, 1:]
    # print(deltas)
    wR = wR.detach().numpy()
    plt.figure()
    N = len(wR)
    time = np.arange(N)
    plt.plot(time, training_outputs[0, :], ".")
    plt.plot(time, wR, ".")
    plt.show()


def run_model():
    dyn_model = Net()
    dyn_model.load_state_dict(torch.load("throttle_model1.pth"))

    N0 = 0
    Nf = -1
    mat = scipy.io.loadmat("throttle_id/transient and braking.mat")
    throttle = mat["throttle"][N0:Nf].T
    dx, N1 = throttle.shape
    wR = mat["wR"][N0:Nf].T
    du, N1 = wR.shape

    time = np.arange(0, len(wR.T)) / 100
    nn_states = np.zeros_like(time)
    state1 = wR[:, 0:1]
    for ii in range(len(time)):
        nn_states[ii] = state1
        input_tensor = torch.from_numpy(
            np.hstack((throttle[:, ii : ii + 1], state1))
        ).float()
        state1 += dyn_model(input_tensor).detach().numpy() * 0.01
    plt.figure()
    plt.plot(time, wR[0, :])
    plt.plot(time, nn_states)
    plt.plot(time, 100 * throttle.T)
    plt.xlabel("time")
    plt.ylabel("wR (rad/s)")
    plt.show()

    N0 = 400
    Nf = -200
    mat = scipy.io.loadmat("results/ltv testing 2021-02-24-19-07-22.mat")
    states = mat["states"][:, ::10]
    controls = mat["inputs"][:, ::10]
    wR = states[4:5, N0 : Nf - 1]
    throttle = controls[1:, N0 : Nf - 1]

    time = np.arange(0, len(wR.T)) / 100
    nn_states = np.zeros_like(time)
    state1 = wR[:, 2:3].copy()
    for ii in range(len(time)):
        nn_states[ii] = state1.copy()
        input_tensor = torch.from_numpy(
            np.hstack((throttle[:, ii : ii + 1], state1))
        ).float()
        state1 += dyn_model(input_tensor).detach().numpy() * 0.01
    plt.figure()
    plt.plot(time[2:], wR[0, 2:])
    plt.plot(time[100:], nn_states[100:])
    # plt.plot(time, throttle.T)
    plt.xlabel("time")
    plt.ylabel("wR (rad/s)")
    plt.show()
    start = 100
    unloaded = nn_states[200:]
    loaded = wR[0, 200:]
    factors = loaded / unloaded
    plt.figure()
    plt.plot(time[200:], factors)
    plt.xlabel("time")
    plt.ylabel("loaded wR / unloaded wR")
    plt.show()
    factor = np.mean(factors)
    print(factor)
    plt.figure()
    plt.plot(time[2:], wR[0, 2:])
    plt.plot(time[200:], (nn_states[200:] * factor))
    plt.xlabel("time")
    plt.ylabel("wR (rad/s)")
    plt.show()

    N0 = 400
    Nf = -200
    mat = scipy.io.loadmat("mppi_data/mppi_states_controls3.mat")
    N0 = 1 * 100
    Nf = -1 * 100
    mat = scipy.io.loadmat("results/ltv testing 2021-02-24-19-07-22.mat")
    states = mat["states"][:, ::10]
    controls = mat["inputs"][:, ::10]
    wR = states[4:5, N0 : Nf - 1]
    throttle = controls[1:, N0 : Nf - 1]

    time = np.arange(0, len(wR.T)) / 100
    nn_states = np.zeros_like(time)
    state1 = wR[:, 2:3].copy()
    for ii in range(len(time)):
        nn_states[ii] = state1.copy()
        input_tensor = torch.from_numpy(
            np.hstack((throttle[:, ii : ii + 1], state1 / factor))
        ).float()
        state1 += dyn_model(input_tensor).detach().numpy() * 0.01
    plt.figure()
    plt.plot(time[2:], wR[0, 2:])
    plt.plot(time[100:], nn_states[100:])
    # plt.plot(time, throttle.T)
    plt.xlabel("time")
    plt.ylabel("wR (rad/s)")
    plt.show()
    # start = 100
    # unloaded = nn_states[200:]
    # loaded = wR[0, 200:]
    # print(params)
    # plt.figure()
    # plt.plot(time[2:], wR[0, 2:])
    # plt.plot(time[200:], (nn_states[200:]))
    # plt.show()


def update_dynamics(state, input, nn=None):
    m_Vehicle_m = 21.7562  # 1270
    m_Vehicle_Iz = 1.124  # 2000
    m_Vehicle_lF = 0.34  # 1.015
    lFR = 0.57  # 3.02
    m_Vehicle_lR = lFR - m_Vehicle_lF
    m_Vehicle_IwF = 0.1  # 8
    m_Vehicle_IwR = 0.0373
    m_Vehicle_rF = 0.095  # 0.325
    m_Vehicle_rR = 0.090  # 0.325
    m_Vehicle_h = 0.12  # .54
    m_g = 9.80665

    tire_B = 4.0  # 10
    tire_C = 1.0
    tire_D = 1.0
    tire_E = 1.0
    tire_Sh = 0.0
    tire_Sv = 0.0

    N, dx = state.shape
    m_nu = 1

    vx = state[:, 0]
    vy = state[:, 1]
    wz = state[:, 2]
    wF = state[:, 3]
    wR = state[:, 4]
    psi = state[:, 5]
    X = state[:, 6]
    Y = state[:, 7]

    m_Vehicle_kSteering = -0.24
    m_Vehicle_cSteering = -0.010
    throttle_factor = 0.38
    # delta = input[:, 0]
    steering = input[0, 0]
    delta = m_Vehicle_kSteering * steering + m_Vehicle_cSteering
    T = input[:, 1]

    min_velo = 0.1
    deltaT = 0.01

    beta = atan2(vy, vx)

    V = sqrt(vx * vx + vy * vy)
    vFx = V * cos(beta - delta) + wz * m_Vehicle_lF * sin(delta)
    vFy = V * sin(beta - delta) + wz * m_Vehicle_lF * cos(delta)
    vRx = vx
    vRy = vy - wz * m_Vehicle_lR

    sEF = -(vFx - wF * m_Vehicle_rF) / (vFx) + tire_Sh
    muFx = (
        tire_D
        * sin(
            tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))
        )
        + tire_Sv
    )
    sEF = -(vRx - wR * m_Vehicle_rR) / (vRx) + tire_Sh
    muRx = (
        tire_D
        * sin(
            tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))
        )
        + tire_Sv
    )

    sEF = atan(vFy / abs(vFx)) + tire_Sh
    alpha = -sEF
    muFy = (
        -tire_D
        * sin(
            tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))
        )
        + tire_Sv
    )
    sEF = atan(vRy / abs(vRx)) + tire_Sh
    alphaR = -sEF
    muRy = (
        -tire_D
        * sin(
            tire_C * atan(tire_B * sEF - tire_E * (tire_B * sEF - atan(tire_B * sEF)))
        )
        + tire_Sv
    )

    fFz = (
        m_Vehicle_m
        * m_g
        * (m_Vehicle_lR - m_Vehicle_h * muRx)
        / (
            m_Vehicle_lF
            + m_Vehicle_lR
            + m_Vehicle_h * (muFx * cos(delta) - muFy * sin(delta) - muRx)
        )
    )
    # fFz = m_Vehicle_m * m_g * (m_Vehicle_lR / 0.57)
    fRz = m_Vehicle_m * m_g - fFz

    fFx = fFz * muFx
    fRx = fRz * muRx
    fFy = fFz * muFy
    fRy = fRz * muRy

    ax = (fFx * cos(delta) - fFy * sin(delta) + fRx) / m_Vehicle_m + vy * wz

    dot_X = cos(psi) * vx - sin(psi) * vy
    dot_Y = sin(psi) * vx + cos(psi) * vy

    next_state = zeros_like(state)
    next_state[:, 0] = vx + deltaT * (
        (fFx * cos(delta) - fFy * sin(delta) + fRx) / m_Vehicle_m + vy * wz
    )
    next_state[:, 1] = vy + deltaT * (
        (fFx * sin(delta) + fFy * cos(delta) + fRy) / m_Vehicle_m - vx * wz
    )
    next_state[:, 1] = vy + deltaT * (
        (fFx * sin(delta) + fFy * cos(delta) + fRy) / m_Vehicle_m - vx * wz
    )
    next_state[:, 2] = (
        wz
        + deltaT
        * ((fFy * cos(delta) + fFx * sin(delta)) * m_Vehicle_lF - fRy * m_Vehicle_lR)
        / m_Vehicle_Iz
    )
    next_state[:, 3] = wF - deltaT * m_Vehicle_rF / m_Vehicle_IwF * fFx
    input_tensor = torch.from_numpy(np.hstack((T, wR / throttle_factor))).float()
    next_state[:, 4] = wR + deltaT * (nn(input_tensor).detach().numpy())
    next_state[:, 5] = psi + deltaT * wz
    next_state[:, 6] = X + deltaT * dot_X
    next_state[:, 7] = Y + deltaT * dot_Y

    # print(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4)

    return next_state


def add_labels():
    plt.subplot(4, 2, 1)
    plt.gca().legend(("Measured vx", "Model vx", "NN vx"))
    plt.xlabel("t (s)")
    plt.ylabel("m/s")
    plt.subplot(4, 2, 2)
    plt.gca().legend(("Measured vy", "Model vy", "NN vy"))
    plt.xlabel("t (s)")
    plt.ylabel("m/s")
    plt.subplot(4, 2, 3)
    plt.gca().legend(("Measured wz", "Model wz", "NN wz"))
    plt.xlabel("t (s)")
    plt.ylabel("rad/s")
    plt.subplot(4, 2, 4)
    plt.gca().legend(("Measured wF", "Model wF", "NN wF"))
    plt.xlabel("t (s)")
    plt.ylabel("rad/s")
    plt.subplot(4, 2, 5)
    plt.gca().legend(("Measured wR", "Model wR", "NN wF"))
    plt.xlabel("t (s)")
    plt.ylabel("rad/s")
    plt.subplot(4, 2, 6)
    plt.gca().legend(("Measured Yaw", "Model Yaw", "NN Yaw"))
    plt.xlabel("t (s)")
    plt.ylabel("rad")
    plt.subplot(4, 2, 7)
    plt.gca().legend(("Measured X", "Model X", "NN X"))
    plt.xlabel("t (s)")
    plt.ylabel("m")
    plt.subplot(4, 2, 8)
    plt.gca().legend(("Measured Y", "Model Y", "NN Y"))
    plt.xlabel("t (s)")
    plt.ylabel("m")


def run_full_model():
    dyn_model = Net()
    dyn_model.load_state_dict(torch.load("throttle_model1.pth"))

    N0 = 45 * 100
    Nf = -5 * 100

    mat = scipy.io.loadmat("results/ltv testing 2021_02_21 18_57_51.mat")
    N0 = 5 * 100
    Nf = -10 * 100

    mat = scipy.io.loadmat("results/ltv testing 2021-02-24-19-07-22.mat")
    states = mat["states"][:, ::10].T
    controls = mat["inputs"][:, ::10].T

    # mat = scipy.io.loadmat('mppi_data/mppi_ff_training1.mat')
    # # measured_states1 = mat['ff'][6:, N0:Nf]
    # # dx, N1 = measured_states1.shape
    # controls1 = mat['ff'][:, N0:Nf].T
    # print(controls1.shape)
    # # du, N1 = controls1.shape
    # # forces = mat['ff'][6:, N0:Nf].T

    states = states[N0 : Nf - 1, :]
    controls = np.hstack(
        (controls[N0 - 11 : Nf - 1 - 11, 0:1], controls[N0 : Nf - 1, 1:2])
    )
    print(controls.shape)
    time = np.arange(0, len(states)) * 0.01
    analytic_states = np.zeros_like(states)
    nn_states = np.zeros_like(states)
    state1 = states[0:1, :]
    state2 = states[0:1, :]
    for ii in range(len(time)):
        analytic_states[ii, :] = state1
        nn_states[ii, :] = state2
        # state1 = update_dynamics(state1, controls[ii:ii+1, :])
        state2 = update_dynamics(state2, controls[ii : ii + 1, :], dyn_model)
        # state1[:, 4:5] = states[ii, 4:5]
        # state2[:, 4:5] = states[ii, 4:5]
        # state2[:, 0] = states[ii, 0]
        # state1[:, 0:2] = states[ii, 0:2]
        # state2[:, 0:2] = states[ii, 0:2]
    plt.figure()
    for ii in range(states.shape[1]):
        plt.subplot(4, 2, ii + 1)
        plt.plot(time, states[:, ii])
        # plt.plot(time, analytic_states[:, ii])
        plt.plot(time, nn_states[:, ii])
    add_labels()
    plt.show()


def trace_model():
    dyn_model = Net()
    dyn_model.load_state_dict(torch.load("throttle_model1.pth"))
    sample = torch.zeros((1, 2))
    sample[0, 0] = 0.5
    sample[0, 1] = 50
    traced_script_module = torch.jit.trace(dyn_model, sample)
    # traced_script_module.save('traced_throttle_model.pt')
    traced_script_module(sample)
    t0 = time.time()
    dyn_model = dyn_model.eval()
    for ii in range(10):
        sample = torch.randn((8 * 20 * 5, 2))
        dyn_model(sample)
        # torch.matmul(torch.randn((1,20)), torch.sigmoid(torch.matmul(torch.randn(20,2), sample)))
    t1 = time.time()
    print(t1 - t0)


if __name__ == "__main__":
    # train_model()
    # run_model()
    run_full_model()
    # trace_model()
