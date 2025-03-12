import torch
from torchmetrics.functional import pairwise_cosine_similarity


def l1(x):
    return torch.cdist(x, x, p=1)


def l2(x):
    return torch.cdist(x, x, p=2)


def cosine(x):
    return 1 - pairwise_cosine_similarity(x)


def poincare(x, c=1.0):
    return _dist(x.unsqueeze(0), x.unsqueeze(1), c)


def artanh(x):
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _mobius_add(x, y, c):
    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c**2 * x2 * y2
    return num / denom


def _dist(x, y, c, keepdim=False):
    c = torch.as_tensor(c).type_as(x)
    sqrt_c = c**0.5
    mobius_result = _mobius_add(-x, y, c).norm(dim=-1, p=2, keepdim=keepdim)
    dist_c = artanh(sqrt_c * mobius_result)
    return dist_c * 2 / sqrt_c
