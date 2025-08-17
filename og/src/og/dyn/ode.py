from typing import Callable, Protocol, TypeVar

import numpy as np

from og.tree_utils import tree_add, tree_inner_product, tree_mac

_State = TypeVar("_State")


class XDotFn(Protocol):
    def __call__(self, state: _State) -> _State:
        ...


def ode2(xdot: XDotFn, dt: float, state: _State) -> _State:
    """Heun's method."""
    state0 = state
    k0 = xdot(state0)

    state1 = tree_mac(state0, dt, k0)
    k1 = xdot(state1)

    # state1 = state0 + dt * (k0 + k1) / 2.0
    coefs = np.array([0.5, 0.5]) * dt
    state_out = tree_add(state0, tree_inner_product(coefs, [k0, k1]))
    return state_out


def ode3(xdot: XDotFn, dt: float, state: _State) -> _State:
    """Bogacki-Shampine method."""
    state0 = state
    k0 = xdot(state0)

    state1 = tree_mac(state0, 0.5 * dt, k0)
    k1 = xdot(state1)

    state2 = tree_mac(state0, 0.75 * dt, k1)
    k2 = xdot(state2)

    # state1 = state0 + dt * (2.0 * k0 + 3.0 * k1 + 4.0 * k2) / 9.0
    coefs = np.array([2.0, 3.0, 4.0]) * dt / 9.0
    state_out = tree_add(state0, tree_inner_product(coefs, [k0, k1, k2]))
    return state_out


def ode4(xdot: XDotFn, dt: float, state: _State) -> _State:
    """RK4."""
    state0 = state
    k0 = xdot(state0)

    state1 = tree_mac(state0, 0.5 * dt, k0)
    k1 = xdot(state1)

    state2 = tree_mac(state0, 0.5 * dt, k1)
    k2 = xdot(state2)

    state3 = tree_mac(state0, dt, k2)
    k3 = xdot(state3)

    # state1 = state0 + dt * (k0 + 2.0 * k1 + 2.0 * k2 + k3) / 6.0
    coefs = np.array([1.0, 2.0, 2.0, 1.0]) * dt / 6.0
    state_out = tree_add(state0, tree_inner_product(coefs, [k0, k1, k2, k3]))
    return state_out
