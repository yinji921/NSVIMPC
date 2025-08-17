from typing import TypeVar

import jax.numpy as jnp
import numpy as np

from og.jax_types import Shape
from og.none import get_or

_Arr = TypeVar("_Arr", np.ndarray, jnp.ndarray, bool)


def as_shape(shape: int | Shape) -> Shape:
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, list):
        shape = tuple(shape)
    if not isinstance(shape, tuple):
        raise ValueError(f"Expected shape {shape} to be a tuple!")
    return shape


def assert_shape(arr: _Arr, shape: int | Shape, label: str | None = None) -> _Arr:
    shape = as_shape(shape)
    label = get_or(label, "array")
    if arr.shape != shape:
        raise AssertionError(f"Expected {label} of shape {shape}, but got shape {arr.shape} of type {type(arr)}!")
    return arr


def assert_scalar(arr: _Arr, label: str | None = None) -> _Arr:
    label = get_or(label, "scalar")
    is_scalar = isinstance(arr, float) or arr.shape == tuple()
    if not is_scalar:
        raise AssertionError(f"Expected {label} but got shape {arr.shape} of type {type(arr)}!")
    return arr
