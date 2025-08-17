from typing import Any, Callable, Iterable, ParamSpec, Sequence, TypeVar

import einops as ei
import jax._src.dtypes
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax._src.lib import xla_client as xc
from jaxtyping import Float

from og.jax_types import Arr

_PyTree = TypeVar("_PyTree")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_Fn = Callable[_P, _R]
# pycharm type inference doesn't seem to work too well when using _Fn with class methods.
_F = TypeVar("_F", bound=Callable)


def jax_use_double():
    jax.config.update("jax_enable_x64", True)


def jax_default_x32():
    jax.config.update("jax_default_dtype_bits", "32")
    reset_default_types()


def reset_default_types():
    is_32 = jax.config.jax_default_dtype_bits == "32"
    jax._src.dtypes.int_ = np.int32 if is_32 else np.int64
    jax._src.dtypes.uint = np.uint32 if is_32 else np.uint64
    jax._src.dtypes.float_ = np.float32 if is_32 else np.float64
    jax._src.dtypes.complex_ = np.complex64 if is_32 else np.complex128
    jax._src.dtypes._default_types = {
        "b": jax._src.dtypes.bool_,
        "i": jax._src.dtypes.int_,
        "u": jax._src.dtypes.uint,
        "f": jax._src.dtypes.float_,
        "c": jax._src.dtypes.complex_,
    }
    dtypes = jax._src.dtypes

    jax.numpy.int_ = jnp.int32 if dtypes.int_ == np.int32 else jnp.int64
    jax.numpy.uint = jnp.uint32 if dtypes.uint == np.uint32 else jnp.uint64
    jax.numpy.float_ = jnp.float32 if dtypes.float_ == np.float32 else jnp.float64
    jax.numpy.complex_ = jnp.complex64 if dtypes.complex_ == np.complex64 else jnp.complex128


def get_cpu_device(idx: int = 0):
    return jax.devices("cpu")[idx]


def get_gpu_device(idx: int = 0):
    return jax.devices("gpu")[idx]


def jax_use_cpu() -> None:
    ctx = jax.default_device(get_cpu_device())
    ctx.__enter__()


def jax2np(pytree: _PyTree) -> _PyTree:
    return jtu.tree_map(np.array, pytree)


def merge01(x):
    return ei.rearrange(x, "n1 n2 ... -> (n1 n2) ...")


def rep_vmap(fn: _F, rep: int, in_axes: int | Sequence[Any] = 0, **kwargs) -> _F:
    for ii in range(rep):
        fn = jax.vmap(fn, in_axes=in_axes, **kwargs)
    return fn


def jax_vmap(fn: _F, in_axes: int | Sequence[Any] = 0, out_axes: Any = 0, rep: int = None) -> _F:
    if rep is not None:
        return rep_vmap(fn, rep=rep, in_axes=in_axes, out_axes=out_axes)

    return jax.vmap(fn, in_axes, out_axes)


def jax_jit_np(
    fn: _F,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    donate_argnums: int | Sequence[int] = (),
    device: xc.Device = None,
    *args,
    **kwargs,
) -> _F:
    jit_fn = jax.jit(fn, static_argnums, static_argnames, donate_argnums, device, *args, **kwargs)

    def wrapper(*args, **kwargs):
        return jax2np(jit_fn(*args, **kwargs))

    return wrapper


def concat_at_front(arr1: Float[Arr, "nx"], arr2: Float[Arr, "T nx"], axis: int = 0) -> Float[Arr, "Tp1 nx"]:
    """
    :param arr1: (nx, )
    :param arr2: (T, nx)
    :param axis: Which axis for arr1 to concat under.
    :return: (T + 1, nx) with [arr1 arr2]
    """
    # The shapes of arr1 and arr2 should be the same without the dim at axis for arr1.
    arr2_shape = list(arr2.shape)
    del arr2_shape[axis]
    assert np.all(np.array(arr1.shape) == np.array(arr2_shape))

    return jnp.concatenate([jnp.expand_dims(arr1, axis=axis), arr2], axis=axis)


def concat_at_end(arr1: Float[Arr, "T nx"], arr2: Float[Arr, "nx"], axis: int = 0) -> Float[Arr, "Tp1 nx"]:
    """
    :param arr1: (T, nx)
    :param arr2: (nx, )
    :param axis: Which axis for arr1 to concat under.
    :return: (T + 1, nx) with [arr1 arr2]
    """
    # The shapes of arr1 and arr2 should be the same without the dim at axis for arr1.
    arr1_shape = list(arr1.shape)
    del arr1_shape[axis]
    assert np.all(np.array(arr1_shape) == np.array(arr2.shape))

    return jnp.concatenate([arr1, jnp.expand_dims(arr2, axis=axis)], axis=axis)


def sinc(x: Float[Arr, "..."]) -> Float[Arr, "..."]:
    """Note: The derivative is not correct at x = 0 because we use where."""
    return jnp.where(x == 0.0, jnp.ones_like(x), jnp.sin(x) / x)
