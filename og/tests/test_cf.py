import jax
import numpy as np

from og.dyn.cf import CF


def test_thrust_acc_conversion():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    rng = np.random.default_rng(seed=12345)

    cf = CF()

    for ii in range(10):
        thrust0 = rng.uniform(0.0, 1.0, size=(4,))
        acc = cf.thrust_to_acc(thrust0)
        thrust1 = cf.acc_to_thrust(acc)

        np.testing.assert_allclose(thrust0, thrust1, rtol=1e-6, atol=1e-6)

    for ii in range(10):
        acc0 = rng.uniform(0.0, 1.0, size=(4,))
        thrust = cf.acc_to_thrust(acc0)
        acc1 = cf.thrust_to_acc(thrust)

        np.testing.assert_allclose(acc0, acc1, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_thrust_acc_conversion()
