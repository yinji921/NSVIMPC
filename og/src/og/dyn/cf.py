import jax.numpy as jnp
import numpy as np
from jaxtyping import Float

from og.jax_types import Arr

# Thrust of motor 1 2 3 4.
Thrust = Float[Arr, "4"]

# [w p q r]. Body frame accelerations.
BodyAcc = Float[Arr, "4"]


class CF:
    def __init__(self):
        self.m = 0.0299
        self.Ixx = 1.395 * 10 ** (-5)
        self.Iyy = 1.395 * 10 ** (-5)
        self.Izz = 2.173 * 10 ** (-5)
        self.CT = 3.1582 * 10 ** (-10)
        self.CD = 7.9379 * 10 ** (-12)
        self.d = 0.03973

        self.normalize_by_CT = True

    def thrust_to_acc(self, thrust: Thrust) -> BodyAcc:
        CT, CD = self.CT, self.CD
        if self.normalize_by_CT:
            CT, CD = 1, CD / CT

        # Try and avoid catastrophic cancellation by doing the sums early
        assert thrust.shape == (4,)
        w_term = jnp.sum(thrust)
        p_term = jnp.sum(thrust * jnp.array([-1.0, -1.0, 1.0, 1.0]))
        q_term = jnp.sum(thrust * jnp.array([-1.0, 1.0, 1.0, -1.0]))
        r_term = jnp.sum(thrust * jnp.array([-1.0, 1.0, -1.0, 1.0]))

        w_dot = CT * w_term / self.m
        p_dot = CT * np.sqrt(2) * self.d * p_term / self.Ixx
        q_dot = CT * np.sqrt(2) * self.d * q_term / self.Iyy
        r_dot = CD * r_term / self.Izz

        return jnp.array([w_dot, p_dot, q_dot, r_dot])

    def acc_to_thrust(self, acc: BodyAcc) -> Thrust:
        CT, CD = self.CT, self.CD
        if self.normalize_by_CT:
            CT, CD = 1, CD / CT

        dw, dp, dq, dr = acc
        # Convert to unnormalized acc.
        wterm = dw * self.m / CT
        pterm = dp * self.Ixx / (CT * np.sqrt(2) * self.d)
        qterm = dq * self.Iyy / (CT * np.sqrt(2) * self.d)
        rterm = dr * self.Izz / CD

        # Solve for thrusts.
        u1 = (wterm - pterm - qterm - rterm) / 4
        u2 = (wterm - pterm + qterm + rterm) / 4
        u3 = (wterm + pterm + qterm - rterm) / 4
        u4 = (wterm + pterm - qterm + rterm) / 4

        return jnp.array([u1, u2, u3, u4])
