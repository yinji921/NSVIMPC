import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
from loguru import logger

from og.jax_types import FloatScalar


def weighted_sum_dict(loss_dict: dict[str, FloatScalar], weights_dict: dict[str, FloatScalar]) -> FloatScalar:
    total_loss = 0
    has_loss = False
    for k, weight in weights_dict.items():
        # Make sure all keys in weights_dict are in loss_dict.
        if k not in loss_dict:
            logger.warning("Weight dict key {} not in loss dict! {}".format(k, loss_dict.keys()))
            continue

        total_loss = total_loss + weight * loss_dict[k]
        has_loss = True

    if not has_loss:
        raise ValueError(
            "No losses found! loss_dict: {}, weights_dict: {}".format(loss_dict.keys(), weights_dict.keys())
        )

    return total_loss


def smoothmax(x, alpha=1e-2, axis: int = 0, bound: str = "upper"):
    out = jnn.logsumexp(x / alpha, axis=axis) * alpha
    if bound == "lower":
        n_els = x.shape[axis]
        out = out - jnp.log(n_els)
        return out
    elif bound == "upper":
        return out
    else:
        raise ValueError("bound must be 'upper' or 'lower'")


def smoothmin(x, axis: int = 0, alpha=1e-2, bound: str = "lower"):
    bound_ = "upper" if bound == "lower" else "upper"
    return -smoothmax(-x, alpha, axis, bound_)


def clip_values_only(x, x_lo, x_hi):
    """Clip values of x between x_lo and x_hi, but keep the gradient of x."""
    diff = lax.stop_gradient(jnp.clip(x, x_lo, x_hi) - x)
    return x + diff


def safe_norm(x, axis: int = None, eps: float = 1e-5):
    # return jnp.linalg.norm(x, axis=axis)
    return jnp.sqrt(jnp.sum(x**2, axis=axis) + eps)
