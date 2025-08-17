import math

import jax.numpy as jnp
import numpy as np
import optax
from attrs import asdict, define

from og.jax_types import FloatScalar, IntScalar


@define
class Schedule:
    def as_dict(self):
        return {"type": type(self).__name__, **asdict(self)}

    @property
    def total_steps(self) -> int:
        return 0

    def make(self) -> optax.Schedule:
        ...


def as_schedule(val: Schedule | float | int) -> Schedule:
    if isinstance(val, Schedule):
        return val

    return Constant(val)


@define
class Constant(Schedule):
    value: float
    steps: int = 0

    @property
    def total_steps(self):
        return self.steps

    def make(self) -> optax.Schedule:
        return optax.constant_schedule(self.value)


@define
class LinDecay(Schedule):
    init: float
    decay_ratio: float
    warmup_steps: int
    trans_steps: int

    def make(self):
        return linear_with_warmup(self.init, self.init / self.decay_ratio, self.warmup_steps, self.trans_steps)


def linear_with_warmup(init_value: float, end_value: float, warmup_steps: int, transition_steps: int) -> optax.Schedule:
    warmup_sched = optax.constant_schedule(init_value)
    linear_sched = optax.linear_schedule(init_value, end_value, transition_steps)

    return optax.join_schedules([warmup_sched, linear_sched], [warmup_steps])
