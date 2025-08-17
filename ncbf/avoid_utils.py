import einops as ei
import ipdb
import jax.debug as jd
import jax.lax as lax
import jax.numpy as jnp
from og.dyn_types import HFloat, TBool, THFloat
from og.shape_utils import assert_shape


def get_disc_avoid(disc_gamma: float, Th_h: THFloat, Th_Vh_dsc: THFloat) -> HFloat:
    """Gets the discounted avoid at the initial timestep only."""

    def loop(h_Vh_, inp):
        h_h, h_Vh_dsc = inp
        h_Vh_ = jnp.maximum(h_h, (1 - disc_gamma) * h_Vh_dsc + disc_gamma * h_Vh_)
        return h_Vh_, None

    T, nh = Th_h.shape
    h_Vh_init = jnp.full(nh, -1e8)
    h_Vh, _ = lax.scan(loop, h_Vh_init, (Th_h, Th_Vh_dsc), reverse=True)
    return h_Vh


def get_max_mc(disc_gamma: float, Th_h: THFloat, Th_Vh_dsc: THFloat):
    def loop(h_Vh_, inp):
        h_h, h_Vh_dsc = inp
        h_Vh_ = jnp.maximum(h_h, (1 - disc_gamma) * h_Vh_dsc + disc_gamma * h_Vh_)
        return h_Vh_, h_Vh_

    T, nh = Th_h.shape
    h_Vh_init = jnp.full(nh, -1e8)
    h_Vh_0, Th_Vh = lax.scan(loop, h_Vh_init, (Th_h, Th_Vh_dsc), reverse=True)
    return Th_Vh


def get_max_gae(disc_gamma: float, gae_lambda: float, Th_h: THFloat, Tp1h_Vh: THFloat, Th_Vh_dsc: THFloat):
    def loop(carry, inp):
        (ii, h_h, h_Vh, h_Vh_dsc) = inp
        Th_Vh_row_next, gae_coefs = carry
        mask = jnp.arange(T) < ii
        mask_h = mask[:, None]

        Th_Vh_row_next_dsc = (1 - disc_gamma) * h_Vh_dsc + disc_gamma * Th_Vh_row_next
        Th_Vh_row = assert_shape(mask_h * jnp.maximum(h_h, Th_Vh_row_next_dsc), [T, nh])

        Th_Vh_row = Th_Vh_row.at[ii + 1, :].set(h_Vh)

        normed_gae_coefs = assert_shape(gae_coefs / gae_coefs.sum(), (T,))
        h_Vh_gae = assert_shape(ei.einsum(Th_Vh_row, normed_gae_coefs, "T nh, T -> nh"), nh)

        # Update GAE coeffs. [1] -> [λ 1] -> [λ² λ 1]
        gae_coefs = jnp.roll(gae_coefs, 1)
        gae_coefs = gae_coefs.at[0].set(gae_coefs[1] * gae_lambda)

        return (Th_Vh_row, gae_coefs), h_Vh_gae

    T, nh = Th_h.shape
    assert Th_h.shape == Th_Vh_dsc.shape
    assert Tp1h_Vh.shape == (T + 1, nh)
    Th_Vh = Tp1h_Vh[:-1, :]

    gae_coefs_init = jnp.zeros(T)
    gae_coefs_init = gae_coefs_init.at[0].set(1.0)

    Th_Vh_row_init = jnp.zeros((T, nh)).at[0, :].set(Tp1h_Vh[-1, :])
    carry_init = (Th_Vh_row_init, gae_coefs_init)

    ts = jnp.arange(T)[::-1]
    inps = (ts, Th_h, Th_Vh, Th_Vh_dsc)

    _, Th_Vh_gae = lax.scan(loop, carry_init, inps, reverse=True)
    return assert_shape(Th_Vh_gae, (T, nh))


def get_max_gae_term(
    disc_gamma: float, gae_lambda: float, Th_h: THFloat, Tp1h_Vh: THFloat, Th_Vh_dsc: THFloat, T_isterm: TBool
):
    def loop(carry, inp):
        (ii, h_h, h_Vh, h_Vh_dsc, is_term) = inp
        Th_Vh_row_next, gae_coefs = carry
        mask = jnp.arange(T) <= ii
        mask_h = mask[:, None]

        Th_Vh_row_next_dsc = (1 - disc_gamma) * h_Vh_dsc + disc_gamma * Th_Vh_row_next
        Th_Vh_row = jnp.maximum(h_h, Th_Vh_row_next_dsc)
        # If this is a terminal state, then h_Vh = h_h.
        Th_Vh_row = jnp.where(is_term, h_h, Th_Vh_row)
        Th_Vh_row = assert_shape(mask_h * Th_Vh_row, [T, nh])

        # jd.print(" ------ ii = {}, isterm: {} -----", ii, is_term)
        # jd.print("h_h: {}", h_h.flatten())
        # jd.print("Th_Vh_row: {}", Th_Vh_row.flatten())

        normed_gae_coefs = assert_shape(gae_coefs / gae_coefs.sum(), (T,))
        h_Vh_gae = assert_shape(ei.einsum(Th_Vh_row, normed_gae_coefs, "T nh, T -> nh"), nh)

        Th_Vh_row_next = Th_Vh_row.at[ii + 1, :].set(h_Vh)

        # Update GAE coeffs. [1] -> [λ 1] -> [λ² λ 1]
        gae_coefs = jnp.roll(gae_coefs, 1)
        gae_coefs = gae_coefs.at[0].set(gae_coefs[1] * gae_lambda)

        return (Th_Vh_row_next, gae_coefs), h_Vh_gae

    T, nh = Th_h.shape
    assert Th_h.shape == Th_Vh_dsc.shape
    assert Tp1h_Vh.shape == (T + 1, nh)
    assert T_isterm.shape == (T,)
    Th_Vh = Tp1h_Vh[:-1, :]

    gae_coefs_init = jnp.zeros(T)
    gae_coefs_init = gae_coefs_init.at[0].set(1.0)

    Th_Vh_row_init = jnp.zeros((T, nh)).at[0, :].set(Tp1h_Vh[-1, :])
    carry_init = (Th_Vh_row_init, gae_coefs_init)

    ts = jnp.arange(T)[::-1]
    inps = (ts, Th_h, Th_Vh, Th_Vh_dsc, T_isterm)
    # ipdb.set_trace()
    _, Th_Vh_gae = lax.scan(loop, carry_init, inps, reverse=True)
    return assert_shape(Th_Vh_gae, (T, nh))
