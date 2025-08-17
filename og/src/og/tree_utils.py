from typing import TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
# import jumpy.numpy as jp  # Replaced with JAX numpy
import numpy as np
from jax import tree_util as jtu

from og.jax_types import Arr, Shape
from og.jax_utils import _PyTree, concat_at_front

_PyTree = TypeVar("_PyTree")


def tree_mac(accum: _PyTree, scalar: float, other: _PyTree, strict: bool = True) -> _PyTree:
    """Tree multiply and accumulate. Return accum + scalar * other, but for pytree.
    :rtype: object
    """

    def mac_inner(a, o):
        if strict:
            assert a.shape == o.shape
        return a + scalar * o

    return jtu.tree_map(mac_inner, accum, other)


def tree_add(t1: _PyTree, t2: _PyTree):
    return jtu.tree_map(lambda a, b: a + b, t1, t2)


def tree_inner_product(coefs: list[float], trees: list[_PyTree]) -> _PyTree:
    def tree_inner_product_(*arrs_):
        arrs_ = list(arrs_)
        out = sum([c * a for c, a in zip(coefs, arrs_)])
        return out

    assert len(coefs) == len(trees)
    return jtu.tree_map(tree_inner_product_, *trees)


def tree_split_dims(tree: _PyTree, new_dims: Shape) -> _PyTree:
    prod_dims = np.prod(new_dims)

    def tree_split_dims_inner(arr: Arr) -> Arr:
        assert arr.shape[0] == prod_dims
        target_shape = new_dims + arr.shape[1:]
        return arr.reshape(target_shape)

    return jtu.tree_map(tree_split_dims_inner, tree)


def tree_concat_at_front(tree1: _PyTree, tree2: _PyTree) -> _PyTree:
    return jtu.tree_map(concat_at_front, tree1, tree2)


def tree_stack(trees: list[_PyTree], axis: int = 0) -> _PyTree:
    def tree_stack_inner(*arrs):
        arrs = list(arrs)
        return jnp.stack(arrs, axis=axis)

    return jtu.tree_map(tree_stack_inner, *trees)


def tree_cat(trees: list[_PyTree], axis: int = 0) -> _PyTree:
    def tree_cat_inner(*arrs):
        arrs = list(arrs)
        return jnp.concatenate(arrs, axis=axis)

    return jtu.tree_map(tree_cat_inner, *trees)


def tree_where(cond, x_tree: _PyTree, y_tree: _PyTree) -> _PyTree:
    def tree_where_inner(x, y):
        return jnp.where(cond, x, y)

    return jax.tree_map(tree_where_inner, x_tree, y_tree)


def tree_copy(tree: _PyTree) -> _PyTree:
    return jax.tree_map(lambda x: x.copy(), tree)


def tree_len(tree: _PyTree) -> int:
    return tree_shape(tree, axis=0)


def tree_shape(tree: _PyTree, axis: int) -> int:
    leaves, treedef = jtu.tree_flatten(tree)
    return leaves[0].shape[axis]
