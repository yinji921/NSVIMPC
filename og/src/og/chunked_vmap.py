from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar

import einops as ei
import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from flax import struct
from jaxtyping import Array, Shaped

from og.jax_types import FloatScalar
from og.jax_utils import merge01
from og.none import get_or
from og.rng import PRNGKey

_P = ParamSpec("_P")
_R = TypeVar("_R")
_Fn = Callable[_P, _R]


def chunked_vmap(fn: _Fn, n_chunks: int) -> _Fn:
    """Assumes that the first argument is to be chunked"""
    vmap_fn = jax.vmap(fn)

    def chunked_fn(b_tree: Shaped[Array, "b *"]):
        b = jtu.tree_leaves(b_tree)[0].shape[0]
        # b = b_arr.shape[0]
        assert b % n_chunks == 0
        chunk_size = b // n_chunks

        mc_tree = jtu.tree_map(lambda b_arr: ei.rearrange(b_arr, "(m c) ... -> m c ...", m=n_chunks, c=chunk_size), b_tree)

        def body(_, c_inp):
            c_output = vmap_fn(c_inp)
            return None, c_output

        _, mc_output = lax.scan(body, None, mc_tree, length=n_chunks)
        b_output = jtu.tree_map(merge01, mc_output)
        return b_output

    return chunked_fn
