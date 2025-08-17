from typing import NamedTuple

import jax.numpy as jnp

from og.dyn.dyn import ControlAffineCtsDyn, Params
from og.dyn_types import State
from og.jax_types import FloatScalar


class DBInt(ControlAffineCtsDyn):
    """
    state = [ p, v ],   control = [ a ]
    """

    NX = 2
    NU = 1

    def f(self, state: State, params: Params = None) -> State:
        p, v = self.chk_x(state)
        return jnp.array([v, 0.0])

    def G(self, state: State, params: Params = None):
        self.chk_x(state)
        G = jnp.array([[0.0], [1.0]])
        assert G.shape == (self.nx, self.nu)
        return G

    def Gu(self, state: State, control: State, params: Params = None) -> State:
        self.chk_x(state)
        (a,) = self.chk_u(control)
        return jnp.array([0.0, a])

    @property
    def x_labels(self) -> list[str]:
        return [r"$p$", r"$v$"]

    @property
    def u_labels(self) -> list[str]:
        return [r"$a$"]


class DBIntForce(ControlAffineCtsDyn):
    """
    state = [ p, v ],   control = [ f ].  a = f / mass.
    """

    class _Params(NamedTuple):
        mass: FloatScalar

    Params = _Params

    NX = 2
    NU = 1

    def f(self, state: State, params: _Params = None) -> State:
        p, v = self.chk_x(state)
        return jnp.array([v, 0.0])

    def G(self, state: State, params: _Params = None):
        self.chk_x(state)
        G = jnp.array([[0.0], [1.0 / params.mass]])
        assert G.shape == (self.nx, self.nu)
        return G

    def Gu(self, state: State, control: State, params: _Params = None) -> State:
        self.chk_x(state)
        (f,) = self.chk_u(control)
        a = f / params.mass
        return jnp.array([0.0, a])

    @property
    def x_labels(self) -> list[str]:
        return [r"$p$", r"$v$"]

    @property
    def u_labels(self) -> list[str]:
        return [r"$f$"]
