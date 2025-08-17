"""Train the noise VAE."""
import shutil

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from flax.training import orbax_utils
from robot_planning.environment.dynamics.noise_modelling.jax_noise_modelling.state_conditioned_noise_model import (
    StateConditionedVAE,
)
from tqdm import tqdm


def load_disturbance_and_state_data(
    data_file_paths: list[str],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load the disturbance and state data from the specified file paths.

    Args:
        data_file_paths: The file paths to load the data from.

    Returns:
        The disturbance and state data.
    """
    # Create arrays to store the data
    disturbances = jnp.zeros((8, 1))
    states = jnp.zeros((11, 1))

    # Load the data
    for data_file_path in data_file_paths:
        data = jnp.load(data_file_path)

        disturbances = jnp.hstack((disturbances, data["disturbances"]))
        states = jnp.hstack((states, data["states"])) # trim out map coords

    return disturbances, states


if __name__ == "__main__":
    prng_key = jax.random.PRNGKey(0)
    checkpoint_path = "/home/ji/SSD/DCSL/Autorally_MPPI_CLBF/robot_planning/experiments/Autorally_experiments/state_dependent_noise_data/vae_checkpoints/19"
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(checkpoint_path)
    batch_size = 32
    validation_split = 0.2
    # Load the data
    data_file_paths = [
        # f"robot_planning/experiments/Autorally_experiments/state_dependent_noise_data/{i}/cbf_n_traj_100_horizon_20_alpha_0.9_Qepsi_0.0_Qey_20.0_log.npz"
        f"/home/ji/SSD/DCSL/Autorally_MPPI_CLBF/robot_planning/experiments/Autorally_experiments/state_dependent_noise_data/{i}/cbf_n_traj_100_horizon_20_alpha_0.9_Qepsi_0.0_Qey_20.0_log.npz"
        # f"/home/ji/SSD/DCSL/Autorally_MPPI_CLBF/robot_planning/experiments/Autorally_experiments/constant_noise_test/{i}/cbf_n_traj_100_horizon_20_alpha_0.9_Qepsi_0.0_Qey_20.0_log.npz"
        for i in range(30)
    ]
    disturbances, states = load_disturbance_and_state_data(data_file_paths)
    # Find indices for center and boundary states
    indices_for_boundary_states = (states[-2, :] > 0.5) | (states[-2, :] < -0.5)
    indices_for_center_states = (states[-2, :] < 0.5) & (states[-2, :] > -0.5)
    boundary_disturbances = disturbances[:, indices_for_boundary_states]
    center_disturbances = disturbances[:, indices_for_center_states]
    # print("boundary_disturbances_shape: ", boundary_disturbances.shape)
    boundary_states = states[:, indices_for_boundary_states]
    boundary_disturbances_std = jnp.std(boundary_disturbances, axis=1, keepdims=True)
    print("boundary_disturbances_variance: ", boundary_disturbances_std **2)
    center_states = states[:, indices_for_center_states]
    center_disturbances_std = jnp.std(center_disturbances, axis=1, keepdims=True)


    # print(indices_for_boundary_states)
    disturbances_mean = jnp.mean(disturbances, axis=1, keepdims=True)
    disturbances_std = jnp.std(disturbances, axis=1, keepdims=True)
    print("overall disturbances_variance: ", disturbances_std**2)
    disturbances = (disturbances - disturbances_mean) / disturbances_std
    states_mean = jnp.mean(states, axis=1, keepdims=True)
    states_std = jnp.std(states, axis=1, keepdims=True)
    states = (states - states_mean) / states_std

    boundary_disturbances = (boundary_disturbances - disturbances_mean)/ disturbances_std
    boundary_states = (boundary_states - states_mean)/states_std
    center_disturbances = (center_disturbances - disturbances_mean)/ disturbances_std
    center_states = (center_states - states_mean)/states_std


    num_batches = disturbances.shape[0] // batch_size
    num_training_batches = int((1 - validation_split) * num_batches)

    model = StateConditionedVAE()
    val_data = jax.tree_map(
        lambda x: x[num_training_batches * batch_size :], (disturbances, states)
    )
    boundary_val_data = jax.tree_map(
        lambda x: x[num_training_batches * batch_size :], (boundary_disturbances, boundary_states)
    )
    center_val_data = jax.tree_map(
        lambda x: x[num_training_batches * batch_size :], (center_disturbances, center_states)
    )

    val_data = (val_data[0].T, val_data[1].T)
    boundary_val_data = (boundary_val_data[0].T, boundary_val_data[1].T)
    center_val_data = (center_val_data[0].T, center_val_data[1].T)

    keys = jax.random.split(prng_key, val_data[0].shape[0])
    boundary_keys = jax.random.split(prng_key, boundary_val_data[0].shape[0])
    center_keys = jax.random.split(prng_key, center_val_data[0].shape[0])

    params = raw_restored['params']
    # print(keys.shape, val_data[0].shape, val_data[1].shape)
    val_w_recon, val_mean, val_logvar = jax.vmap(model.apply, in_axes=(None, 0, 0, 0))(
        params, keys, val_data[0], val_data[1]
    )
    boundary_val_w_recon, boundary_val_mean, boundary_val_logvar = jax.vmap(model.apply, in_axes=(None, 0, 0, 0))(
        params, boundary_keys, boundary_val_data[0], boundary_val_data[1]
    )
    center_val_w_recon, center_val_mean, center_val_logvar = jax.vmap(model.apply, in_axes=(None, 0, 0, 0))(
        params, center_keys, center_val_data[0], center_val_data[1]
    )

    disturbances_mean = disturbances_mean.T
    disturbances_std = disturbances_std.T
    val_w_recon = (val_w_recon * disturbances_std) + disturbances_mean
    center_val_w_recon = (center_val_w_recon * disturbances_std) + disturbances_mean
    boundary_val_w_recon = (boundary_val_w_recon * disturbances_std) + disturbances_mean

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4)
    # for i in range(2):
    #     for j in range(4):
    #         axs[i, j].hist(
    #             val_data[0][:, 4 * i + j], bins=100, label="True", color="r", alpha=0.5
    #         )
    #         print(f"variance of {4*i+j} dimension ", jnp.var(val_data[0][:, 4*i+j]))
    #         axs[i, j].hist(
    #             val_w_recon[:, 4 * i + j],
    #             bins=100,
    #             label="Reconstructed",
    #             color="b",
    #             alpha=0.5,
    #         )
    # for i in range(2):
    #     for j in range(4):
    #         axs[i, j].hist(
    #             boundary_val_data[0][:, 4 * i + j], bins=100, label="True", color="r", alpha=0.5
    #         )
    #         print(f"variance of {4*i+j} dimension ", jnp.var(boundary_val_data[0][:, 4*i+j]))
    #         axs[i, j].hist(
    #             boundary_val_w_recon[:, 4 * i + j],
    #             bins=100,
    #             label="Reconstructed",
    #             color="b",
    #             alpha=0.5,
    #         )

    for i in range(2):
        for j in range(4):
            # axs[i, j].hist(
            #     val_w_recon[:, 4 * i + j], bins=100, label="Entire Distribution", color="r", alpha=0.5
            # )
            print(f"variance of {4 * i + j} dimension for center states ", jnp.var(center_val_w_recon[:, 4 * i + j]))
            print(f"variance of {4*i+j} dimension for boundary states", jnp.var(boundary_val_w_recon[:, 4*i+j]))
            axs[i, j].hist(boundary_val_w_recon[:, 4 * i + j],bins=100,label="Boundary Distribution",color="b",alpha=0.5,
            )
            axs[i, j].hist(
                center_val_w_recon[:, 4 * i + j], bins=100, label="Center Distribution", color="g", alpha=0.5
            )
    plt.legend()
    plt.show()