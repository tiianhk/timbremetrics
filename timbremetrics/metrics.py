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
    to_target_length,
)
from .distances import l1, l2, dot_product, cosine, poincare


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
        device: The device to load data. Default: CPU
        dtype: The data type of audio input. Default: torch.float32
            Sometimes it needs to be the same as the model parameters'.
        use_fadtk_model: Whether to use a fadtk model.
        fadtk_audio_loader: The dedicated audio loader for the fadtk model.
            If use_fadtk_model is True, this must be provided.
        fadtk_time_average: Whether to average the embeddings over time.
            For non-fadtk models, if time averaging is needed, implement it in the model.
        datasets: A list of dataset names to evaluate the model on.
            If None, all datasets are used.
        distances: A list of distance functions to compute distance
            matrices. Default: [l2, cosine]
        metrics: A list of metric functions to evaluate the model's representations.
            Default: [mae, ndcg_retrieve_sim, spearman_corr, kendall_corr, triplet_agreement]
        sample_rate: The sample rate to use for audio loading. If None,
            the original sample rate (44100 Hz) is used. If use_fadtk_model is True,
            this must be provided.
        pad_to_the_longer_length: Whether to pad the shorter audio signals to the
            length of the longer one in a pair. If True, fixed_duration must be None.
        fixed_duration: The duration to pad or truncate audio signals to.
            If provided, pad_to_the_longer_length must be False.
    """

    def __init__(
        self,
        device=None,
        dtype=None,
        use_fadtk_model=False,
        fadtk_audio_loader=None,
        fadtk_time_average=True,
        datasets=None,
        distances=None,
        metrics=None,
        sample_rate=None,
        pad_to_the_longer_length=False,
        fixed_duration=None,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.use_fadtk_model = use_fadtk_model
        self.fadtk_audio_loader = fadtk_audio_loader
        if self.use_fadtk_model:
            assert self.fadtk_audio_loader is not None, (
                "A dedicated audio loader needs to be provided. "
                "Use fadtk_model.load_wav"
            )
        self.fadtk_time_average = fadtk_time_average

        self.datasets = datasets
        if self.datasets is None:
            self.datasets = list_datasets()
        else:
            assert set(self.datasets).issubset(
                set(list_datasets())
            ), f"Invalid dataset in {self.datasets}. Choose from {list_datasets()}"

        self.distances = [l2, cosine]
        if distances is not None:
            assert set(distances).issubset(set(self.distances)), (
                f"Invalid distance function in {distances}. "
                "Choose from [l1, l2, dot_product, cosine, poincare]"
            )
            self.distances = distances

        self.metrics = [
            mae,
            ndcg_retrieve_sim,
            spearman_corr,
            kendall_corr,
            triplet_agreement,
        ]
        if metrics is not None:
            assert set(metrics).issubset(
                set(self.metrics)
            ), f"Invalid metric function in {metrics}. Choose from {self.metrics}"
            self.metrics = metrics

        self.sample_rate = sample_rate
        if self.use_fadtk_model:
            assert self.sample_rate is not None, (
                "Sample rate used by the fadtk model needs to be provided. "
                "Use fadtk_model.sr"
            )
        self.pad_to_the_longer_length = pad_to_the_longer_length
        self.fixed_duration = fixed_duration

        audio_loader = AudioLoader(
            datasets=self.datasets,
            device=self.device,
            dtype=self.dtype,
            fadtk_audio_loader=self.fadtk_audio_loader,
            target_sr=self.sample_rate,
            fixed_duration=self.fixed_duration,
        )
        self.audio_datasets = audio_loader._load_audio_datasets()
        self.true_dissim = get_true_dissim(datasets=self.datasets, device=self.device)
        self.num_pairs, self.num_stimuli = self._retrieve_dissim_info()

    def _retrieve_dissim_info(self):
        num_pairs = torch.tensor(0.0).to(self.device)
        num_stimuli = torch.tensor(0.0).to(self.device)
        for d in self.datasets:
            N = self.true_dissim[d].shape[0]
            num_stimuli += N
            num_pairs += N * (N - 1) / 2
        return num_pairs, num_stimuli

    def forward(self, model):
        """Compute the evaluation metrics for a model."""
        if hasattr(model, "training"):
            if model.training:
                model.eval()
        pred_dissim = {dist_fn.__name__: {} for dist_fn in self.distances}
        for d in self.datasets:
            if self.pad_to_the_longer_length:
                embeddings = self._extract_dataset_paired_embeddings(
                    model, self.audio_datasets[d]
                )
            else:
                embeddings = self._extract_dataset_embeddings(
                    model, self.audio_datasets[d]
                )
            for dist_fn in self.distances:
                dist = dist_fn(
                    embeddings, paired_embeddings=self.pad_to_the_longer_length
                )
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
                if model.name in [
                    "clap-2023",
                    "clap-laion-audio",
                    "clap-laion-music",
                    "cdpam-acoustic",
                    "cdpam-content",
                ]:  # context windows > max audio length, only take the first frame
                    embedding = embedding[0]
                if self.fadtk_time_average:
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

    @torch.no_grad()
    def _extract_dataset_paired_embeddings(self, model, dataset: list):
        embeddings = {}
        embeddings["num_stimuli"] = len(dataset)
        for i in range(len(dataset)):
            for j in range(i + 1, len(dataset)):
                audio_i = dataset[i]["audio"]
                audio_j = dataset[j]["audio"]
                max_len = max(audio_i.shape[-1], audio_j.shape[-1])
                audio_i = to_target_length(audio_i, max_len)
                audio_j = to_target_length(audio_j, max_len)
                if self.use_fadtk_model:
                    embedding_i = model.get_embedding(audio_i)
                    embedding_j = model.get_embedding(audio_j)
                    if not isinstance(embedding_i, Tensor):
                        embedding_i = torch.tensor(embedding_i)
                    if not isinstance(embedding_j, Tensor):
                        embedding_j = torch.tensor(embedding_j)
                else:
                    embedding_i = model(audio_i)
                    embedding_j = model(audio_j)
                    assert isinstance(
                        embedding_i, Tensor
                    ), "Model must return a torch.Tensor."
                    assert isinstance(
                        embedding_j, Tensor
                    ), "Model must return a torch.Tensor."
                embedding_i = embedding_i.flatten().float()
                embedding_j = embedding_j.flatten().float()
                embeddings[(i, j)] = (embedding_i, embedding_j)
        return embeddings

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
