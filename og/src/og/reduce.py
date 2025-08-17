from typing import Any, Callable, Literal, Protocol, Sequence, Tuple, Union

import jax.numpy as jnp

from og.jax_types import AnyFloat

ReduceLiteral = Literal["mean", "min", "max", "sum"]


class ReduceFn(Protocol):
    def __call__(self, arr: AnyFloat, axis: int | None = None) -> AnyFloat:
        ...


def get_reduce_fn(reduce_str: ReduceLiteral) -> ReduceFn:
    reduce_dict: dict[Literal, ReduceFn] = dict(
        mean=jnp.mean,
        min=jnp.min,
        max=jnp.max,
        sum=jnp.sum,
    )
    return reduce_dict[reduce_str]
