from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
import timbremetrics.pmath as pmath


def compute_from_paired_embeddings(x: dict, dist_fn: Callable):
    N = x["num_stimuli"]
    dist_mtx = torch.zeros(N, N, device=x[(0, 1)][0].device)
    for i in range(N):
        for j in range(i + 1, N):
            dist_mtx[i, j] = dist_fn(x[(i, j)][0], x[(i, j)][1])
    return dist_mtx


def l1(x, paired_embeddings=False):
    if paired_embeddings == False:
        return torch.cdist(x, x, p=1)
    else:
        dist_fn = lambda x, y: torch.norm(x - y, p=1)
        return compute_from_paired_embeddings(x, dist_fn)


def l2(x, paired_embeddings=False):
    if paired_embeddings == False:
        return torch.cdist(x, x, p=2)
    else:
        dist_fn = lambda x, y: torch.norm(x - y, p=2)
        return compute_from_paired_embeddings(x, dist_fn)


def dot_product(x, paired_embeddings=False):
    if paired_embeddings == False:
        return -x @ x.T
    else:
        dist_fn = lambda x, y: -torch.dot(x, y)
        return compute_from_paired_embeddings(x, dist_fn)


def cosine(x, paired_embeddings=False):
    if paired_embeddings == False:
        return 1 - pairwise_cosine_similarity(x)
    else:
        dist_fn = lambda x, y: 1 - F.cosine_similarity(x, y, dim=0)
        return compute_from_paired_embeddings(x, dist_fn)


def poincare(x, c=1.0, paired_embeddings=False):
    if paired_embeddings == False:
        projector = ToPoincare(c)
        x_poincare = projector(x)
        return pmath.dist_matrix(x_poincare, x_poincare, c=c)
    else:
        raise NotImplementedError("Not implemented yet.")


"""
code below is from https://github.com/leymir/hyperbolic-image-embeddings/blob/master/hyptorch/nn.py
MIT License
Copyright (c) 2019 Valentin Khrulkov
"""


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None, riemannian=True):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(
                torch.Tensor(
                    [
                        c,
                    ]
                )
            )
        else:
            self.c = c

        self.train_x = train_x

        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c

        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):

        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c))
        return self.grad_fix(pmath.project(pmath.expmap0(x, c=self.c), c=self.c))

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)
