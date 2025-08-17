from typing import NamedTuple

from og.jax_types import AnyFloat


class MeanStd(NamedTuple):
    mean: AnyFloat
    std: AnyFloat

    def normalize(self, unnorm: AnyFloat) -> AnyFloat:
        return (unnorm - self.mean) / self.std

    def unnormalize(self, norm: AnyFloat) -> AnyFloat:
        return norm * self.std + self.mean
