from typing import TypeVar

import jax
import jax.numpy as jnp

from og.dyn_types import Control, State
from og.shape_utils import assert_shape

Params = TypeVar("Params")


class CtsDyn:
    NX = None
    NU = None

    def xdot(self, state: State, control: Control, params: Params = None) -> State:
        raise NotImplementedError("")

    @property
    def nx(self) -> int:
        return self.NX

    @property
    def nu(self) -> int:
        return self.NU

    @property
    def x_labels(self) -> list[str]:
        raise NotImplementedError("")

    @property
    def u_labels(self) -> list[str]:
        raise NotImplementedError("")

    @property
    def name(self):
        return self.__class__.__name__

    def chk_x(self, state: State):
        return assert_shape(state, self.nx, "state")

    def chk_u(self, control: Control):
        return assert_shape(control, self.nu, "control")


class ControlAffineCtsDyn(CtsDyn):
    def f(self, state: State, params: Params = None) -> State:
        raise NotImplementedError("")

    def Gu(self, state: State, control: Control, params: Params = None):
        raise NotImplementedError("")

    def G(self, state: State, params: Params = None):
        control = jnp.zeros(self.nu)
        return jax.jacobian(self.Gu)(state, control, params)

    def xdot(self, state: State, control: Control, params: Params = None) -> State:
        self.chk_x(state)
        self.chk_u(control)
        f = self.chk_x(self.f(state, params=params))
        Gu = self.chk_x(self.Gu(state, control, params=params))
        dx = f + Gu
        return self.chk_x(dx)
