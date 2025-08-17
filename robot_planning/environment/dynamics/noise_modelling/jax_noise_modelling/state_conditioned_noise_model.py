"""Defines a state-conditioned noise model for use in risk-aware planning."""
import flax.linen as nn
import jax
import jax.numpy as jnp
import orbax.checkpoint


class StateConditionedEncoder(nn.Module):
    latent_dim: int = 8
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, w: jnp.ndarray, x: jnp.ndarray):
        """Computes the output of the encoder.

        Args:
            w: The input data.
            x: The state.
        """
        w = jnp.concatenate([w, x], axis=-1)
        w = nn.Dense(self.hidden_dim, name="fc1")(w)
        w = nn.relu(w)
        w = nn.Dense(self.hidden_dim, name="fc2")(w)
        w = nn.relu(w)
        mean = nn.Dense(self.latent_dim, name="fc3_mean")(w)
        logvar = nn.Dense(self.latent_dim, name="fc3_logvar")(w)

        return mean, logvar


class StateConditionedDecoder(nn.Module):
    hidden_dim: int = 32
    output_dim: int = 8

    @nn.compact
    def __call__(self, w: jnp.ndarray, x: jnp.ndarray):
        """Computes the output of the decoder.

        Args:
            w: The latent variable.
            x: The state.
        """
        w = jnp.concatenate([w, x], axis=-1)
        w = nn.Dense(self.hidden_dim, name="fc1")(w)
        w = nn.relu(w)
        w = nn.Dense(self.hidden_dim, name="fc2")(w)
        w = nn.relu(w)
        w = nn.Dense(self.output_dim, name="fc3")(w)

        return w


class StateConditionedVAE(nn.Module):
    """A state-conditioned VAE for modeling noise."""

    latent_dim: int = 32
    hidden_dim: int = 32
    output_dim: int = 8

    def setup(self):
        self.encoder = StateConditionedEncoder(
            latent_dim=self.latent_dim, hidden_dim=self.hidden_dim
        )
        self.decoder = StateConditionedDecoder(
            hidden_dim=self.hidden_dim, output_dim=self.output_dim
        )

    def __call__(self, prng_key, w: jnp.ndarray, x: jnp.ndarray):
        """Computes the output of the VAE.

        Args:
            prng_key: The PRNG key.
            w: The input data.
            x: The state.
        """
        mean, logvar = self.encoder(w, x)
        logvar = 1e-2 * logvar
        z = self.reparameterize(prng_key, mean, logvar)
        w_recon = self.decoder(z, x)

        return w_recon, mean, logvar

    def reparameterize(self, prng_key, mean, logvar):
        """Reparameterizes the latent variable."""
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(prng_key, mean.shape)
        return mean + eps * std

    def load_params_from_checkpoint(self, checkpoint_path: str):
        """Loads the parameters from a checkpoint."""
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = orbax_checkpointer.restore(checkpoint_path)
        return checkpoint["params"]



