import torch

import torch.autograd as autograd


def projected_sliced_score_matching(score_net, samples, n_particles=1):
    dup_samples = (
        samples.unsqueeze(0)
        .expand(n_particles, *samples.shape)
        .contiguous()
        .view(-1, *samples.shape[1:])
    )
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]

    grad1 = grad1.view(dup_samples.shape[0], -1)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.0

    loss2 = torch.sum((vectors * grad2).view(dup_samples.shape[0], -1), dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()


def anneal_dsm_score_estimation(
    scorenet, samples, sigmas, labels=None, anneal_power=2.0, n_particles=1
):
    if labels is None:
        labels = torch.randint(
            0, len(sigmas), (samples.shape[0],), device=samples.device
        )

    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    dup_samples = (
        perturbed_samples.unsqueeze(0)
        .expand(n_particles, *samples.shape)
        .contiguous()
        .view(-1, *samples.shape[1:])
    )
    dup_labels = (
        labels.unsqueeze(0).expand(n_particles, *labels.shape).contiguous().view(-1, 1)
    )
    dup_samples.requires_grad_(True)

    # use Rademacher
    vectors = torch.randn_like(dup_samples)

    grad1 = scorenet(dup_samples, dup_labels)
    gradv = torch.sum(grad1 * vectors)
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]

    grad1 = grad1.view(dup_samples.shape[0], -1)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.0

    loss2 = torch.sum((vectors * grad2).view(dup_samples.shape[0], -1), dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = (loss1 + loss2) * (used_sigmas.squeeze() ** 2)

    return loss.mean(dim=0)

