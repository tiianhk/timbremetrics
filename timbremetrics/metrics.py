from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.functional.retrieval import retrieval_normalized_dcg
from torchmetrics.functional import spearman_corrcoef, kendall_rank_corrcoef

from .utils import (
    AudioLoader,
    get_true_dissim,
    list_datasets,
    mask,
    min_max_normalization,
)
from .distances import l1, l2, cosine, poincare


def mae(pred: Tensor, true: Tensor) -> Tensor:
    return torch.sum(torch.abs(pred - true))


def _prepare_rank_scores(pred: Tensor, true: Tensor) -> tuple[Tensor, Tensor]:
    # symmetrize
    pred = pred + pred.T
    true = true + true.T
    # remove self-comparisons
    mask_ = torch.ones_like(pred, dtype=torch.bool)
    mask_.fill_diagonal_(False)
    N = pred.shape[0]
    pred = pred[mask_].reshape(N, N - 1)
    true = true[mask_].reshape(N, N - 1)
    return pred, true


def ndcg_retrieve_sim(pred: Tensor, true: Tensor) -> Tensor:
    pred, true = _prepare_rank_scores(pred, true)
    # flip to retrieve similar items
    pred = 1 - pred
    true = 1 - true
    ndcg_scores = torch.zeros_like(pred[:, 0])
    # retrieval_normalized_dcg flattens input tensors so we use a loop
    for i in range(pred.shape[0]):
        ndcg_scores[i] = retrieval_normalized_dcg(pred[i], true[i])
    return ndcg_scores.sum()


def spearman_corr(pred: Tensor, true: Tensor) -> Tensor:
    pred, true = _prepare_rank_scores(pred, true)
    return spearman_corrcoef(pred.T, true.T).sum()


def kendall_corr(pred: Tensor, true: Tensor) -> Tensor:
    pred, true = _prepare_rank_scores(pred, true)
    return torch.nansum(kendall_rank_corrcoef(pred.T, true.T))


def _compute_triplet_agreement_one_anchor(
    pred: Tensor, true: Tensor, margin: float = 0.1
) -> Tensor:
    true_diff = true.unsqueeze(0) - true.unsqueeze(1)
    pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
    valid_mask = torch.abs(true_diff) > margin
    upper_triangle_mask = torch.triu(
        torch.ones_like(valid_mask, dtype=bool), diagonal=1
    )
    valid_mask = valid_mask & upper_triangle_mask
    agreement_mask = (pred_diff * true_diff) > 0
    agreements = torch.sum(agreement_mask & valid_mask)
    valid_pairs = torch.sum(valid_mask)
    if valid_pairs == 0:
        return torch.tensor(0.0).to(pred.device)
    return agreements / valid_pairs


def triplet_agreement(pred: Tensor, true: Tensor, margin: float = 0.1) -> Tensor:
    pred, true = _prepare_rank_scores(pred, true)
    scores = torch.zeros_like(pred[:, 0])
    for i in range(pred.shape[0]):
        scores[i] = _compute_triplet_agreement_one_anchor(
            pred[i], true[i], margin=margin
        )
    return scores.sum()


class TimbreMetric(nn.Module):
    """Compute how well a model's embeddings capture perceptual timbre similarity.

    Args:
        model: A model that outputs embeddings for audio signals. (default: None)
        device: The device to use for computation, normally "cuda" or "cpu".
            If model is nn.Module and has parameters, this is inferred. (default: None)
        dtype: The data type to use for computation, normally torch.float32.
            If model is nn.Module and has parameters, this is inferred. (default: None)
        use_fadtk_model: Whether to use a fadtk model. If True,
            model must be provided for suitable audio loading. (default: False)
        fadtk_keep_time_dimension: Whether to keep the time dimension
            of the fadtk model output. (default: False)
        distances: A list of distance functions to use for computing dissimilarity
            matrices. (default: [l1, l2, cosine, poincare])
        metrics: A list of metric functions to use for evaluating the model's performance.
            (default: [mae, ndcg_retrieve_sim, spearman_corr, kendall_corr, triplet_agreement])
        sample_rate: The sample rate to use for audio loading. If None,
            the original sample rate (44100 Hz) is used. (default: None)
        pad_to_max_duration: Whether to pad audio signals to the maximum duration
            in the dataset. If True, fixed_duration must be None. (default: False)
        fixed_duration: The duration to pad or truncate audio signals to.
            If provided, pad_to_max_duration must be False. (default: None)
    """

    def __init__(
        self,
        model=None,
        device=None,
        dtype=None,
        use_fadtk_model=False,
        fadtk_keep_time_dimension=False,
        distances=None,
        metrics=None,
        sample_rate=None,
        pad_to_max_duration=False,
        fixed_duration=None,
    ):
        super().__init__()

        self.model = model
        self.device = device
        self.dtype = dtype
        if isinstance(self.model, nn.Module):
            if any(p.numel() > 0 for p in self.model.parameters()):
                self.device, self.dtype = self._retrieve_model_info(self.model)

        self.use_fadtk_model = use_fadtk_model
        if self.use_fadtk_model:
            assert (
                self.model is not None
            ), "If using FADTK, model must be provided for suitable audio loading."
        self.fadtk_keep_time_dimension = fadtk_keep_time_dimension

        self.distances = [l1, l2, cosine, poincare]
        if distances is not None:
            assert set(distances).issubset(
                set(self.distances)
            ), "Invalid distance function."
            self.distances = distances

        self.metrics = [
            mae,
            ndcg_retrieve_sim,
            spearman_corr,
            kendall_corr,
            triplet_agreement,
        ]
        if metrics is not None:
            assert set(metrics).issubset(set(self.metrics)), "Invalid metric function."
            self.metrics = metrics

        self.sample_rate = sample_rate
        self.pad_to_max_duration = pad_to_max_duration
        self.fixed_duration = fixed_duration
        self.datasets = list_datasets()
        audio_loader = AudioLoader(
            fadtk_model=self.model if self.use_fadtk_model else None,
            device=self.device,
            dtype=self.dtype,
            target_sr=self.sample_rate,
            pad_to_max_duration=self.pad_to_max_duration,
            fixed_duration=self.fixed_duration,
        )
        self.audio_datasets = audio_loader._load_audio_datasets()
        self.true_dissim = get_true_dissim(device=self.device)
        self.num_pairs, self.num_stimuli = self._retrieve_dissim_info()

    def _retrieve_model_info(self, model: nn.Module):
        first_param = next(model.parameters())
        device = first_param.device
        dtype = first_param.dtype
        return device, dtype

    def _retrieve_dissim_info(self):
        num_pairs = torch.tensor(0.0).to(self.device)
        num_stimuli = torch.tensor(0.0).to(self.device)
        for d in self.datasets:
            N = self.true_dissim[d].shape[0]
            num_stimuli += N
            num_pairs += N * (N - 1) / 2
        return num_pairs, num_stimuli

    def forward(self, model=None):
        if model is None:
            assert self.model is not None, "Model must be provided."
            model = self.model
        if hasattr(model, "training"):
            if model.training:
                model.eval()
        pred_dissim = {dist_fn.__name__: {} for dist_fn in self.distances}
        for d in self.datasets:
            embeddings = self._extract_dataset_embeddings(model, self.audio_datasets[d])
            for dist_fn in self.distances:
                dist = dist_fn(embeddings)
                pred_dissim[dist_fn.__name__][d] = min_max_normalization(mask(dist))
        return self._evaluate(pred_dissim)

    @torch.no_grad()
    def _extract_dataset_embeddings(self, model, dataset: list):
        embeddings = []
        for x in dataset:
            audio = x["audio"]
            if self.use_fadtk_model:
                embedding = model.get_embedding(audio)
                if not isinstance(embedding, Tensor):
                    embedding = torch.tensor(embedding)
                if not self.fadtk_keep_time_dimension:
                    embedding = torch.mean(
                        embedding, dim=0
                    )  # (n_frames, n_features) -> (n_features)
            else:
                embedding = model(audio)  # audio shape (1, n_samples)
                assert isinstance(
                    embedding, Tensor
                ), "Model must return a torch.Tensor."
            embedding = embedding.flatten().float()
            embeddings.append(embedding)
        if len(set([embedding.shape for embedding in embeddings])) > 1:
            raise ValueError(
                "The model is outputting embeddings of different shapes. "
                + "All embeddings must have the same shape."
            )
        return torch.stack(embeddings)

    def _evaluate(self, nested_pred_dissim: dict):
        nested_scores = {}
        for dist_fn in self.distances:
            nested_scores[dist_fn.__name__] = {}
            for metric_fn in self.metrics:
                nested_scores[dist_fn.__name__][metric_fn.__name__] = (
                    self._evaluate_one_metric(
                        nested_pred_dissim[dist_fn.__name__], metric_fn
                    )
                )
        return nested_scores

    def _evaluate_one_metric(self, pred_dissim: dict, metric_fn: Callable) -> Tensor:
        score = torch.tensor(0.0).to(self.device)
        for d in self.datasets:
            score += metric_fn(pred_dissim[d], self.true_dissim[d])
        if metric_fn.__name__ == "mae":
            score /= self.num_pairs
        else:
            score /= self.num_stimuli
        return score
