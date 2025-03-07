import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.functional.retrieval import retrieval_normalized_dcg
from torchmetrics.functional import spearman_corrcoef, kendall_rank_corrcoef

from .utils import (
    get_audio,
    get_true_dissim,
    list_datasets,
    mask,
    min_max_normalization,
)
from .distances import euclidean, cosine, poincare


def mae(pred: Tensor, true: Tensor):
    return torch.sum(torch.abs(pred - true))


def rank_based_metric(function, pred: Tensor, true: Tensor, flip=False):
    pred = pred + pred.T
    true = true + true.T
    mask_ = torch.ones_like(pred, dtype=torch.bool)
    mask_.fill_diagonal_(False)
    N = pred.shape[0]
    pred = pred[mask_].reshape(N, N - 1)
    true = true[mask_].reshape(N, N - 1)
    scores = torch.zeros_like(pred[:, 0])
    for i in range(N):
        if flip:
            scores[i] = function(1 - pred[i], 1 - true[i])
        else:
            scores[i] = function(pred[i], true[i])
    return scores.sum()


def ndcg_retrieve_sim(pred: Tensor, true: Tensor):
    return rank_based_metric(retrieval_normalized_dcg, pred, true, flip=True)


def spearman_corr(pred: Tensor, true: Tensor):
    return rank_based_metric(spearman_corrcoef, pred, true)


def kendall_corr(pred: Tensor, true: Tensor):
    return rank_based_metric(kendall_rank_corrcoef, pred, true)


class TimbreMetric(nn.Module):
    def __init__(self, model, distance=cosine, sample_rate=None, fixed_duration=None):
        super().__init__()
        self.model = model
        self.model_id = id(model)
        self.distance = distance
        self.sample_rate = sample_rate
        self.fixed_duration = fixed_duration
        self._retrieve_model_info(model)
        self.datasets = list_datasets()
        self.audio = get_audio(
            device=self.device,
            dtype=self.dtype,
            target_sr=self.sample_rate,
            fixed_duration=self.fixed_duration,
        )
        self.true_dissim = get_true_dissim(device=self.device)
        self._retrieve_dissim_info()

    def _retrieve_model_info(self, model):
        self.device = None
        self.dtype = None
        if isinstance(model, nn.Module):
            if any(model.parameters()):
                first_param = next(model.parameters())
                self.dtype = first_param.dtype
                self.device = first_param.device

    def _retrieve_dissim_info(self):
        self.num_pairs = torch.tensor(0.0).to(self.device)
        self.num_stimuli = torch.tensor(0.0).to(self.device)
        for d in self.datasets:
            N = self.true_dissim[d].shape[0]
            self.num_stimuli += N
            self.num_pairs += N * (N - 1) / 2

    def forward(self, model=None):
        if model is None:
            model = self.model
        assert (
            id(model) == self.model_id
        ), "Model does not match the model used to initialize the metric."
        if hasattr(model, "training"):
            if model.training:
                model.eval()
        pred_dissim = {}
        for d in self.datasets:
            embeddings = self._extract_dataset_embeddings(model, self.audio[d])
            dist = self.distance(embeddings)
            pred_dissim[d] = min_max_normalization(mask(dist))
        return self._evaluate(pred_dissim)

    @torch.no_grad()
    def _extract_dataset_embeddings(self, model, dataset: list):
        embeddings = []
        for x in dataset:
            audio = x["audio"]
            embedding = model(audio)  # audio shape (1, num_samples)
            assert isinstance(embedding, Tensor)
            embedding = embedding.flatten()
            embeddings.append(embedding)
        if len(set([embedding.shape for embedding in embeddings])) > 1:
            raise ValueError(
                "The model is outputting embeddings of different shapes. "
                + "All embeddings must have the same shape."
            )
        return torch.stack(embeddings)

    def _evaluate(self, pred_dissim):
        mae_score = torch.tensor(0.0).to(self.device)
        ndcg_score = torch.tensor(0.0).to(self.device)
        spearman_score = torch.tensor(0.0).to(self.device)
        kendall_score = torch.tensor(0.0).to(self.device)
        for d in self.datasets:
            mae_score += mae(pred_dissim[d], self.true_dissim[d])
            ndcg_score += ndcg_retrieve_sim(pred_dissim[d], self.true_dissim[d])
            spearman_score += spearman_corr(pred_dissim[d], self.true_dissim[d])
            kendall_score += kendall_corr(pred_dissim[d], self.true_dissim[d])
        mae_score /= self.num_pairs
        ndcg_score /= self.num_stimuli
        spearman_score /= self.num_stimuli
        kendall_score /= self.num_stimuli
        return {
            "mae": mae_score,
            "ndcg": ndcg_score,
            "spearman": spearman_score,
            "kendall": kendall_score,
        }
