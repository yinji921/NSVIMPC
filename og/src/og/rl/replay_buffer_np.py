from typing import TypeVar

import einops as ei
import jax
import jax.lax as lax
import jax.random as jr
import numpy as np
from flax import struct
from typing_extensions import Self

from og.jax_types import BoolScalar, IntScalar
from og.rng import PRNGKey
from og.tree_utils import tree_len
from og.treenode_utils import prettynode

Item = TypeVar("Item")
BItem = TypeVar("BItem")


@prettynode
class ReplayBufferNp(struct.PyTreeNode):
    data: BItem
    head: IntScalar
    size: IntScalar
    is_full: BoolScalar
    capacity: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, item_proto: Item, capacity: int) -> Self:
        data = jax.tree_map(lambda x: np.array(ei.repeat(x, "... -> b ...", b=capacity)), item_proto)
        return ReplayBufferNp(data, np.array(0), np.array(0), np.array(False), capacity)

    def push(self, item: Item) -> Self:
        def push_fn(data, arr):
            data[insert_pos] = arr

        insert_pos = self.head
        jax.tree_map(push_fn, self.data, item)
        self.head[...] = (insert_pos + 1) % self.capacity
        self.size[...] = lax.select(self.is_full, self.capacity, self.size + 1)
        self.is_full[...] = self.size == self.capacity
        return self

    def push_batch(self, b_item: BItem, batch_size: int | None = None) -> Self:
        def push_fn(data, arr):
            data[self.head : self.head + size1] = arr[:size1]
            data[:size2] = arr[size1:]

        if batch_size is None:
            batch_size = tree_len(b_item)

        # size2 is the number of items we wrap around from the start.
        size1 = min(batch_size, self.capacity - self.head)
        size2 = batch_size - size1
        jax.tree_map(push_fn, self.data, b_item)

        self.head[...] = (self.head + batch_size) % self.capacity
        self.size[...] = self.capacity if self.is_full else min(self.size + batch_size, self.capacity)
        self.is_full[...] = self.size == self.capacity
        return self

    def get_at_index(self, idx: int | IntScalar) -> Item:
        if np.any(idx >= self.size):
            raise IndexError(f"Trying to index {idx}, size {self.size}")
        return jax.tree_map(lambda x: x[idx], self.data)

    def uniform_sample_np(self, rng: np.random.Generator, batch_size: int) -> BItem:
        assert isinstance(batch_size, int), f"batch_size should be int, got {type(batch_size).__name__}"
        b_idx = rng.integers(0, self.size, (batch_size,))
        return self.get_at_index(b_idx)

    def uniform_sample_jax(self, key: PRNGKey, batch_size: int) -> BItem:
        assert isinstance(batch_size, int), f"batch_size should be int, got {type(batch_size).__name__}"
        b_idx = jr.randint(key, (batch_size,), minval=0, maxval=self.size)
        return jax.jax_vmap(self.get_at_index)(b_idx)
