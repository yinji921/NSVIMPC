from typing import Type

import flax.linen as nn
from og.networks.network_utils import ActFn, default_nn_init


class ValueNet(nn.Module):
    base: Type[nn.Module]
    nh: int
    act: ActFn | None = None

    @nn.compact
    def __call__(self, *args, **kwargs):
        out = self.base()(*args, **kwargs)
        h_pred = nn.Dense(self.nh, kernel_init=default_nn_init())(out)

        if self.act is not None:
            h_pred = self.act(h_pred)

        return h_pred
