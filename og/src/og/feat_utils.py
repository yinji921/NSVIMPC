import jax.numpy as jnp
import numpy as np

from og.jax_types import FloatScalar, Vec2, Vec3, Vec4


def encode_vec2d(vec2: Vec2, eps: float = 1e-5) -> Vec3:
    assert vec2.shape == (2,)
    norm = jnp.sqrt(jnp.sum(vec2**2) + eps)
    vec2_unit = vec2 / norm
    return jnp.array([vec2_unit[0], vec2_unit[1], norm])


def encode_vec2d_exp(vec2: Vec2, eps: float = 1e-5) -> Vec3:
    assert vec2.shape == (2,)
    norm = jnp.sqrt(jnp.sum(vec2**2) + eps)
    vec2_unit = vec2 / norm
    norm_encoded = (jnp.exp(-2 * norm) - 0.15) / 0.15
    return jnp.array([vec2_unit[0], vec2_unit[1], norm_encoded])


def encode_vec2d_log(vec2: Vec2, max_dist: float, eps: float = 1e-5, clip: bool = True) -> Vec3:
    assert vec2.shape == (2,)
    norm = jnp.sqrt(jnp.sum(vec2**2) + eps)
    vec2_unit = vec2 / norm
    norm_encoded = jnp.log(1 + norm)
    # Normalize norm_encoded to be [0, 1] within [0, max_dist].
    norm_encoded = norm_encoded / np.log(1 + max_dist)
    # Normalize norm_encoded to be [-1, 1].
    norm_encoded = norm_encoded * 2 - 1
    if clip:
        # Clip it to prevent OOD.
        norm_encoded = norm_encoded.clip(min=-1, max=1)
    return jnp.array([vec2_unit[0], vec2_unit[1], norm_encoded])


def encode_vec3d(vec3: Vec3, eps: float = 1e-5) -> Vec4:
    assert vec3.shape == (3,)
    norm = jnp.sqrt(jnp.sum(vec3**2) + eps)
    norm_feat = norm - 0.5
    vec3_unit = vec3 / norm
    return jnp.array([vec3_unit[0], vec3_unit[1], vec3_unit[2], norm_feat])


def encode_vec3d_exp(vec3: Vec3, eps: float = 1e-5) -> Vec4:
    assert vec3.shape == (3,)
    norm = jnp.sqrt(jnp.sum(vec3**2) + eps)
    vec3_unit = vec3 / norm
    norm_encoded = (jnp.exp(-2 * norm) - 0.15) / 0.15
    return jnp.array([vec3_unit[0], vec3_unit[1], vec3_unit[2], norm_encoded])


def encode_vec3d_log(vec3: Vec3, max_dist: float, eps: float = 1e-5, clip: bool = True) -> Vec4:
    assert vec3.shape == (3,)
    norm = jnp.sqrt(jnp.sum(vec3**2) + eps)
    vec3_unit = vec3 / norm
    norm_encoded = jnp.log(1 + norm)
    # Normalize norm_encoded to be [0, 1] within [0, max_dist].
    norm_encoded = norm_encoded / np.log(1 + max_dist)
    # Normalize norm_encoded to be [-1, 1].
    norm_encoded = norm_encoded * 2 - 1
    if clip:
        # Clip it to prevent OOD.
        norm_encoded = norm_encoded.clip(min=-1, max=1)
    return jnp.array([vec3_unit[0], vec3_unit[1], vec3_unit[2], norm_encoded])


def encode_exp(x: FloatScalar, maxval: float) -> FloatScalar:
    out = jnp.exp(-4 * x / maxval)
    return (out - 0.4) / 0.5
