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
    # Setup the PRNG key
    prng_key = jax.random.PRNGKey(0)

    # Hyperparams
    validation_split = 0.2
    batch_size = 32
    learning_rate = 1e-3
    kl_weight = 0.0002
    epochs = 20

    # Load the data
    data_file_paths = [
        # f"robot_planning/experiments/Autorally_experiments/state_dependent_noise_data/{i}/cbf_n_traj_100_horizon_20_alpha_0.9_Qepsi_0.0_Qey_20.0_log.npz"
        f"/home/ji/SSD/DCSL/Autorally_MPPI_CLBF/robot_planning/experiments/Autorally_experiments/state_dependent_noise_data/{i}/cbf_n_traj_100_horizon_20_alpha_0.9_Qepsi_0.0_Qey_20.0_log.npz"
        # f"/home/ji/SSD/DCSL/Autorally_MPPI_CLBF/robot_planning/experiments/Autorally_experiments/constant_noise_test/{i}/cbf_n_traj_100_horizon_20_alpha_0.9_Qepsi_0.0_Qey_20.0_log.npz"
        for i in range(30)
    ]
    disturbances, states = load_disturbance_and_state_data(data_file_paths)

    # Normalize the data
    disturbances_mean = jnp.mean(disturbances, axis=1, keepdims=True)
    disturbances_std = jnp.std(disturbances, axis=1, keepdims=True)
    disturbances = (disturbances - disturbances_mean) / disturbances_std
    print("Original disturbance variance: ", disturbances_std ** 2)

    states_mean = jnp.mean(states, axis=1, keepdims=True)
    states_std = jnp.std(states, axis=1, keepdims=True)
    states = (states - states_mean) / states_std

    # Shuffle and transpose so the batch dimension is first
    prng_key, key = jax.random.split(prng_key)
    indices = jax.random.permutation(key, jnp.arange(disturbances.shape[1]))
    disturbances = disturbances[:, indices].T
    states = states[:, indices].T

    # Trim to an integer number of batches and split into training and validation
    num_batches = disturbances.shape[0] // batch_size
    disturbances = disturbances[: num_batches * batch_size]
    states = states[: num_batches * batch_size]
    num_training_batches = int((1 - validation_split) * num_batches)
    num_val_batches = num_batches - num_training_batches
    training_data = jax.tree_map(
        lambda x: x[: num_training_batches * batch_size], (disturbances, states)
    )
    val_data = jax.tree_map(
        lambda x: x[num_training_batches * batch_size :], (disturbances, states)
    )

    # Reshape to add a batch dimension
    training_data = jax.tree_map(
        lambda x: x.reshape(num_training_batches, batch_size, -1), training_data
    )
    val_data = jax.tree_map(
        lambda x: x.reshape(num_val_batches, batch_size, -1), val_data
    )

    print(f"Training data shape: {training_data[0].shape}")
    print(f"Validation data shape: {val_data[0].shape}")

    # Create the model
    model = StateConditionedVAE()
    prng_key, model_key = jax.random.split(prng_key)
    w, x = jax.tree_util.tree_map(lambda x: x[0, 0], training_data)
    params = model.init(model_key, prng_key, w, x)

    # Define loss functions
    def kl_divergence(mean, logvar):
        return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

    def mse_loss(w, w_recon):
        return jnp.mean((w_recon - w) ** 2)

    # Define a batched loss function
    def loss_fn(params, w, x, kl_weight, key):
        keys = jax.random.split(key, w.shape[0])
        w_recon, mean, logvar = jax.vmap(model.apply, in_axes=(None, 0, 0, 0))(
            params, keys, w, x
        )
        mse = jnp.mean(jax.vmap(mse_loss)(w, w_recon))
        kl = jnp.mean(jax.vmap(kl_divergence)(mean, logvar))
        return mse + kl_weight * kl, {"recon_loss": mse, "kl_div": kl}

    # Create the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    # Create a checkpoint manager
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # Delete old checkpoints
    shutil.rmtree(
        # "robot_planning/experiments/Autorally_experiments/state_dependent_noise_data/vae_checkpoints",
        "/home/ji/SSD/DCSL/Autorally_MPPI_CLBF/robot_planning/experiments/Autorally_experiments/state_dependent_noise_data/vae_checkpoints",
        # "/home/ji/SSD/DCSL/Autorally_MPPI_CLBF/robot_planning/experiments/Autorally_experiments/constant_noise_test/vae_checkpoints",
        ignore_errors=True,
    )

    # Training loop
    val_loss_trace = []
    for epoch in range(epochs):
        # Training
        for batch in tqdm(range(num_batches)):
            # Get the batch
            w, x = jax.tree_map(lambda x: x[batch], training_data)

            # Compute the loss and gradients
            prng_key, step_key = jax.random.split(prng_key)
            (loss, _), grads = loss_grad_fn(params, w, x, kl_weight, step_key)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        # Validation
        val_loss = 0.0
        val_losses = {"recon_loss": 0.0, "kl_div": 0.0}
        for batch in range(num_batches):
            # Get the batch
            w, x = jax.tree_map(lambda x: x[batch], val_data)

            # Compute the loss
            prng_key, key = jax.random.split(prng_key)
            loss, aux = jax.jit(loss_fn)(params, w, x, kl_weight, key)
            val_loss += loss
            val_losses = jax.tree_util.tree_map(lambda x, y: x + y, val_losses, aux)

        # Print some info
        print(f"Epoch: {epoch} | Val Loss: {val_loss / num_batches}")
        print(
            f"\tRecon Loss: {val_losses['recon_loss'] / num_batches} | KL Div: {val_losses['kl_div'] / num_batches}"
        )
        val_loss_trace.append(val_loss / num_batches)

        # Save a checkpoint
        if epoch % 1 == 0:
            ckpt = {
                "params": params,
                "disturbance_mean": disturbances_mean,
                "disturbance_std": disturbances_std,
                "state_mean": states_mean,
                "state_std": states_std,
            }
            save_args = orbax_utils.save_args_from_target(ckpt)
            orbax_checkpointer.save(
                # f"robot_planning/experiments/Autorally_experiments/state_dependent_noise_data/vae_checkpoints/{epoch}",
                f"/home/ji/SSD/DCSL/Autorally_MPPI_CLBF/robot_planning/experiments/Autorally_experiments/state_dependent_noise_data/vae_checkpoints/{epoch}",
                # f"/home/ji/SSD/DCSL/Autorally_MPPI_CLBF/robot_planning/experiments/Autorally_experiments/constant_noise_test/vae_checkpoints/{epoch}",
                ckpt,
                save_args=save_args,
            )

    # Get the encodings for all of the validation data
    val_data = jax.tree_map(
        lambda x: x.reshape(num_val_batches * batch_size, -1), val_data
    )
    keys = jax.random.split(prng_key, val_data[0].shape[0])
    print(keys.shape, val_data[0].shape, val_data[1].shape)
    val_w_recon, val_mean, val_logvar = jax.vmap(model.apply, in_axes=(None, 0, 0, 0))(
        params, keys, val_data[0], val_data[1]
    )

    # Plot the true and reconstructed noise
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4)
    for i in range(2):
        for j in range(4):
            axs[i, j].hist(
                val_data[0][:, 4 * i + j], bins=100, label="True", color="r", alpha=0.5
            )
            print(f"variance of {4*i+j} dimension ", jnp.var(val_data[0][:, 4*i+j]))
            axs[i, j].hist(
                val_w_recon[:, 4 * i + j],
                bins=100,
                label="Reconstructed",
                color="b",
                alpha=0.5,
            )
    plt.legend()
    plt.show()
    # Plot the validation loss
    val_loss_trace = jnp.array(val_loss_trace)
    plt.plot(val_loss_trace)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.yscale("log")
    plt.show()
