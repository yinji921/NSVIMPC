from typing import TypeVar

import jax.numpy as jnp
import jax.tree_util as jtu

from og.jax_types import FloatScalar

_PyTree = TypeVar("_PyTree")


def compute_norm(grad: _PyTree) -> _PyTree:
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad: _PyTree, max_norm: float) -> tuple[_PyTree, FloatScalar]:
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm
