import jax.numpy as jnp
import numpy as np

from og.jax_types import FloatScalar, RotMat2D


def wrap_to_pi(theta: FloatScalar) -> FloatScalar:
    """Wrap an angle to [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def wrap_to_pos(theta: FloatScalar) -> FloatScalar:
    """Wrap an angle to [0, 2*pi]."""
    return theta % (2 * np.pi)


def rot2d(theta: FloatScalar) -> RotMat2D:
    if isinstance(theta, jnp.ndarray):
        c, s = jnp.cos(theta), jnp.sin(theta)
        return jnp.array([[c, -s], [s, c]])
    else:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s], [s, c]])
