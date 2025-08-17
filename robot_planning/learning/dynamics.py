import numpy as np
import torch


def langevin_dynamics(score_net, x_init, lr=0.1, steps=1000):
    for i in range(steps):
        current_lr = lr
        x_init = (
            x_init
            + current_lr / 2 * score_net(x_init).detach()
            + torch.randn_like(x_init) * np.sqrt(current_lr)
        )
    return x_init


def anneal_langevin_dynamics(score_net, x_init, sigmas, lr=0.1, n_steps_each=500):
    for sigma in sigmas:
        current_lr = lr * (sigma / max(sigmas)) ** 2
        for i in range(n_steps_each):
            x_init = (
                x_init
                + current_lr
                / 2
                * score_net(x_init, sigma.repeat(x_init.shape[0], 1)).detach()
                + torch.randn_like(x_init) * np.sqrt(current_lr)
            )
    return x_init
