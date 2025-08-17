import math
import numpy as np

import torch


class GMMDistSampling(object):
    def __init__(self, dim, n_mixs=2):
        self.mix_probs = torch.rand(n_mixs)
        self.means = torch.randint(-6, 6, (n_mixs, 2), dtype=torch.float32)
        self.sigma = 1
        self.std = torch.stack(
            [torch.ones(dim) * self.sigma for i in range(len(self.mix_probs))], dim=0
        )

    def sample(self, n):
        n = n[0]
        mix_idx = torch.multinomial(self.mix_probs, n, replacement=True)
        means = self.means[mix_idx]
        stds = self.std[mix_idx]
        return torch.randn_like(means) * stds + means

    def log_prob(self, samples):
        logps = []
        for i in range(len(self.mix_probs)):
            logps.append(
                (
                    -((samples - self.means[i]) ** 2).sum(dim=-1)
                    / (2 * self.sigma ** 2)
                    - 0.5 * np.log(2 * np.pi * self.sigma ** 2)
                )
                + self.mix_probs[i].log()
            )
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp


class ComplexGMMDistSampling(object):
    def __init__(self, dim, n_mixs=4):
        self.mix_probs = torch.rand(n_mixs)
        self.means = torch.tensor(
            [[0, -6], [-6, 0], [6, 0], [0, 6]], dtype=torch.float32
        )

        self.std = torch.rand((n_mixs, 2)) * 3

    def sample(self, n):
        n = n[0]
        mix_idx = torch.multinomial(self.mix_probs, n, replacement=True)
        means = self.means[mix_idx]
        stds = self.std[mix_idx]
        return torch.randn_like(means) * stds + means

    def log_prob(self, samples):
        logps = []
        for i in range(len(self.mix_probs)):
            logps.append(
                (
                    -((samples - self.means[i]) ** 2).sum(dim=-1)
                    / (2 * self.sigma ** 2)
                    - 0.5 * np.log(2 * np.pi * self.sigma ** 2)
                )
                + self.mix_probs[i].log()
            )
        logp = torch.logsumexp(torch.stack(logps, dim=0), dim=0)
        return logp


class SpiralSampling2D(object):
    def sample(self, n):
        if isinstance(n, tuple):
            n = n[0]

        z = torch.randn(n, 2)
        m = torch.sqrt(torch.rand(n // 2)) * 540 * (2 * math.pi) / 360
        d1x = -torch.cos(m) * m + torch.rand(n // 2) * 0.5
        d1y = torch.sin(m) * m + torch.rand(n // 2) * 0.5
        x = torch.cat(
            [torch.stack([d1x, d1y], dim=1), torch.stack([-d1x, -d1y], dim=1)], dim=0
        )
        return x / 1.5 + 0.1 * z

    def log_prob(self, samples):
        raise NotImplementedError
