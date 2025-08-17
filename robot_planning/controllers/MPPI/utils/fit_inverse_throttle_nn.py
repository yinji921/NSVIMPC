import os
import torch
import numpy as np
from robot_planning.environment.dynamics.autorally_dynamics import throttle_model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.optimize import curve_fit


class InverseThrottleModel(torch.nn.Module):
    def __init__(self):
        super(InverseThrottleModel, self).__init__()
        self.cuda = torch.device("cpu")

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 1),
        )

    def load_from_file(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, wR, r_wheel_acceleration):
        input_tensor = torch.hstack((wR / 100, r_wheel_acceleration / 100))

        output = self.nn(input_tensor)
        return output


@torch.no_grad()
def evaluate_throttle_model(throttle_model, T, wR):
    """Evaluate the throttle model with the given throttle and wheel speed

    args:
        throttle_model: the autorally_dynamics.throttle_model to evaluate
        T: throttle commands (n_batch x 1 tensor)
        wR: rear wheel speeds (n_batch x 1 tensor)
    returns:
        rear wheel acceleration
    """
    throttle_factor = 0.45
    input_tensor = torch.hstack((T, wR / throttle_factor))
    r_wheel_acceleration = throttle_model(input_tensor)
    return r_wheel_acceleration


@torch.no_grad()
def generate_training_data(throttle_model, n_pts):
    """Generate triples of (T, wheel speed, wheel acceleration) to train off of"""
    # Sample uniformly for both throttle and wheel speed
    T = torch.Tensor(n_pts, 1).uniform_(0, 1.0)
    wR = torch.Tensor(n_pts, 1).uniform_(0, 100.0)

    # Get the corresponding wheel acceleration
    r_wheel_acceleration = evaluate_throttle_model(throttle_model, T, wR)

    return T, wR, r_wheel_acceleration


def do_train_inverse_model(
    throttle_model, n_pts, n_epochs, learning_rate, batch_size=64
):
    """Train an inverse of the given throttle model

    args:
        throttle_model: the model to invert
        n_pts: the number of training points to use
        n_epochs: the number of epochs to use (same as the # of data presentations)
        learning_rate: step size
    returns:
        the trained inverse model
    """
    # Create the inverse model
    inverse_model = InverseThrottleModel()

    # Generate some training data
    T, wR, r_wheel_acceleration = generate_training_data(throttle_model, n_pts)

    # Make a loss function and optimizer
    mse_loss_fn = torch.nn.MSELoss(reduction="mean")
    l1_loss_fn = torch.nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(inverse_model.parameters(), lr=learning_rate)

    # Optimize in mini-batches
    for epoch in range(n_epochs):
        permutation = torch.randperm(n_pts)

        loss_accumulated = 0.0
        for i in range(0, n_pts, batch_size):
            batch_indices = permutation[i : i + batch_size]

            # Forward pass: predict throttle from wheel speed and acceleration
            T_predicted = inverse_model(
                wR[batch_indices], r_wheel_acceleration[batch_indices]
            )

            # Compute the loss and backpropagate
            loss = mse_loss_fn(T_predicted, T[batch_indices])
            loss += l1_loss_fn(T_predicted, T[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accumulated += loss.detach()

        print(f"Epoch {epoch}: {loss_accumulated / (n_pts / batch_size)}")

    return inverse_model


def load_forward_model():
    # Load the model to invert
    throttle_nn_file_path = (
        "environment/dynamics/autorally_dynamics/throttle_model1.pth"
    )
    throttle_nn = throttle_model.Net()
    throttle_nn.load_state_dict(torch.load(throttle_nn_file_path))

    return throttle_nn


def load_inverse_throttle_model():
    inverse_model_path = (
        os.path.dirname(os.path.realpath(__file__)) + "/inverse_throttle_model.pth"
    )
    inverse_model = InverseThrottleModel()
    inverse_model.load_from_file(inverse_model_path)

    return inverse_model


def train_inverse(throttle_nn, save=True):
    """Train an inverse model of the given throttle model and optionally save it"""

    # Train an inverse model
    n_pts = int(5e5)
    n_epochs = 20
    lr = 1e-2
    inverse_model = do_train_inverse_model(throttle_nn, n_pts, n_epochs, lr)
    if save:
        torch.save(
            inverse_model.state_dict(),
            os.path.dirname(os.path.realpath(__file__)) + "/inverse_throttle_model.pth",
        )

    return inverse_model


def evaluate(forward_model, inverse_model):
    # Plot a comparison with the forward model
    with torch.no_grad():
        _T = np.linspace(0, 1.0, 100)
        _wR = np.linspace(0.0, 100)
        T, wR = np.meshgrid(_T, _wR)
        T_flat = torch.from_numpy(T).reshape(-1, 1).float()
        wR_flat = torch.from_numpy(wR).reshape(-1, 1).float()
        r_wheel_acceleration = evaluate_throttle_model(forward_model, T_flat, wR_flat)
        T_predicted = inverse_model(wR_flat, r_wheel_acceleration)

        ax = plt.axes(projection="3d")
        ax.plot_surface(r_wheel_acceleration.reshape(wR.shape), wR, T, edgecolor="none")
        ax.plot_wireframe(
            r_wheel_acceleration.reshape(wR.shape),
            wR,
            T_predicted.reshape(wR.shape),
            color="black",
        )

        ax.set_xlabel("rear_wheel_acceleration")
        ax.set_ylabel("rear_wheel_speed")
        ax.set_zlabel("throttle")
        plt.show()

        ax = plt.axes(projection="3d")
        ax.plot_surface(
            r_wheel_acceleration.reshape(wR.shape),
            wR,
            (T - T_predicted.reshape(wR.shape).numpy()) ** 2,
            cmap="jet",
            edgecolor="none",
        )

        ax.set_xlabel("rear_wheel_acceleration")
        ax.set_ylabel("rear_wheel_speed")
        ax.set_zlabel("squared model error")
        plt.show()


if __name__ == "__main__":
    forward_model = load_forward_model()
    inverse_model = train_inverse(forward_model, save=True)
    evaluate(forward_model, inverse_model)
