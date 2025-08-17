from typing import Type

import flax.linen as nn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

from og.rng import PRNGKey


class Ensemble(nn.Module):
    net_cls: Type[nn.Module]
    num: int

    @nn.compact
    def __call__(self, *args, in_axes=None):
        ensemble = nn.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=in_axes,
            out_axes=0,
            axis_size=self.num,
        )
        out = ensemble()(*args)
        if isinstance(out, jnp.ndarray):
            assert out.shape[0] == self.num
        return out


def subsample_ensemble(key: PRNGKey, params: dict, num_sample: int, num_qs: int):
    index = jr.choice(key, num_qs, shape=(num_sample,), replace=False)

    # Index into the params for the ensemble.
    ensemble_params = jtu.tree_map(lambda param: param[index], params["params"])

    params = params | {"params": ensemble_params}
    return params
