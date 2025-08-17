from typing import Any, Callable, Literal, Sequence, Tuple, Union

import flax.linen as nn
import jax.lax as lax

from og.jax_types import AnyFloat, FloatScalar, Shape
from og.rng import PRNGKey

ActFn = Callable[[AnyFloat], AnyFloat]

InitFn = Callable[[PRNGKey, Shape, Any], Any]

default_nn_init = nn.initializers.xavier_uniform

HidSizes = Sequence[int]

PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]]


def scaled_init(initializer: nn.initializers.Initializer, scale: float) -> nn.initializers.Initializer:
    def scaled_init_inner(*args, **kwargs) -> AnyFloat:
        return scale * initializer(*args, **kwargs)

    return scaled_init_inner


ActLiteral = Literal["relu", "tanh", "elu", "swish", "silu", "gelu", "softplus", "identity"]


def softplus(x, beta: FloatScalar = 20.0):
    return nn.softplus(beta * x) / beta


def get_act_from_str(act_str: ActLiteral) -> ActFn:
    act_dict: dict[Literal, ActFn] = dict(
        relu=nn.relu,
        tanh=nn.tanh,
        elu=nn.elu,
        swish=nn.swish,
        silu=nn.silu,
        gelu=nn.gelu,
        softplus=softplus,
        identity=lambda x: x,
    )
    return act_dict[act_str]
