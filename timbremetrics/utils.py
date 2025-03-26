from typing import Union
import os
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F

from .paths import STIMULI_DIR, TRUE_DISSIM_DIR


def min_max_normalization(a):
    return (a - a.min()) / (a.max() - a.min())


def mask(x):
    mask = torch.ones_like(x).triu(1)  # upper triangle, diagonal excluded
    return mask * x


def list_datasets():
    dataset_files = [
        f.replace("_dissimilarity_matrix.txt", "") for f in os.listdir(TRUE_DISSIM_DIR)
    ]
    return sorted(dataset_files)


class AudioLoader:
    def __init__(
        self,
        datasets=None,
        device=None,
        dtype=None,
        fadtk_audio_loader=None,
        target_sr=None,
        fixed_duration=None,
    ):
        self.datasets = datasets
        self.device = device
        self.dtype = dtype
        self.fadtk_audio_loader = fadtk_audio_loader
        self.target_sr = target_sr
        self.fixed_duration = fixed_duration

    def _load_audio_datasets(self):
        if self.datasets is None:
            self.datasets = list_datasets()
        else:
            assert set(self.datasets).issubset(
                set(list_datasets())
            ), f"Invalid dataset in {self.datasets}. Choose from {list_datasets()}"
        audio_datasets = {}
        for d in self.datasets:
            audio_datasets[d] = self._load_one_audio_dataset(d)
        return audio_datasets

    def _load_one_audio_dataset(self, dataset):
        audio_files = os.listdir(os.path.join(STIMULI_DIR, dataset))
        audio_files = sorted(audio_files)
        audio_dataset = []
        for audio_file in audio_files:
            if not audio_file.endswith(".aiff"):
                continue
            audio, sr = self._load_one_audio_file(dataset, audio_file)
            audio_dataset.append(
                {"file": audio_file, "audio": audio, "sample_rate": sr}
            )
        return audio_dataset

    def _load_one_audio_file(self, dataset, audio_file):
        f = os.path.join(STIMULI_DIR, dataset, audio_file)
        if self.fadtk_audio_loader is not None:
            audio = self.fadtk_audio_loader(f)
            sr = self.target_sr
        else:
            audio, sr = torchaudio.load(f, backend="soundfile")
            audio = audio.to(device=self.device, dtype=self.dtype)
            if self.target_sr is not None and sr != self.target_sr:
                audio = torchaudio.transforms.Resample(sr, self.target_sr)(audio)
                sr = self.target_sr
        if self.fixed_duration is not None:
            target_sample_num = int(self.fixed_duration * sr)
            audio = to_target_length(audio, target_sample_num)
        return audio, sr


def to_target_length(audio, target_sample_num: int):
    if isinstance(audio, torch.Tensor):
        if audio.shape[-1] > target_sample_num:
            audio = audio[..., :target_sample_num]
        elif audio.shape[-1] < target_sample_num:
            padding = target_sample_num - audio.shape[-1]
            audio = F.pad(audio, (0, padding))
    elif isinstance(audio, np.ndarray):
        if audio.shape[-1] > target_sample_num:
            audio = audio[..., :target_sample_num]
        elif audio.shape[-1] < target_sample_num:
            padding = target_sample_num - audio.shape[-1]
            pad_width = [(0, 0)] * (audio.ndim - 1) + [(0, padding)]
            audio = np.pad(audio, pad_width)
    else:
        # for descript audio codec
        from audiotools import AudioSignal

        assert isinstance(audio, AudioSignal)
        assert isinstance(audio.audio_data, torch.Tensor)
        if audio.audio_data.shape[-1] > target_sample_num:
            audio.audio_data = audio.audio_data[..., :target_sample_num]
        elif audio.audio_data.shape[-1] < target_sample_num:
            padding = target_sample_num - audio.audio_data.shape[-1]
            audio.audio_data = F.pad(audio.audio_data, (0, padding))
    return audio


def get_true_dissim(datasets=None, device=None):
    if datasets is None:
        datasets = list_datasets()
    else:
        assert set(datasets).issubset(set(list_datasets())), "Invalid dataset."
    true_dissim = {}
    for d in datasets:
        f = os.path.join(TRUE_DISSIM_DIR, f"{d}_dissimilarity_matrix.txt")
        data = torch.tensor(np.loadtxt(f)).to(device)
        true_dissim[d] = min_max_normalization(mask(data))
    return true_dissim


def print_results(model_name, results):
    print(f"{model_name}:")
    for distance, metrics in results.items():
        print(f"    {distance}:")
        for metric, value in metrics.items():
            print(f"        {metric}: {value.item():.3f}")
    print()


def write_results_to_yaml(fn, model_name, results):
    for k, v in results.items():
        results[k] = {kk: round(vv.item(), 3) for kk, vv in v.items()}
    import yaml

    try:
        with open(fn, "r") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}

    data[model_name] = results
    with open(fn, "w") as f:
        yaml.dump(data, f)


def get_datasets_weights(datasets: list):
    assert set(datasets).issubset(set(list_datasets())), "Invalid dataset."
    weights = {"mae": 0, "rank_based": 0}
    true_dissim = get_true_dissim()
    for d in datasets:
        N = true_dissim[d].shape[0]
        weights["mae"] += N * (N - 1) / 2
        weights["rank_based"] += N
        print(f"{d}: {N} stimuli, {N * (N - 1) / 2} pairs")
    return weights


def merge_metrics(
    metrics_list: list[dict[str, Union[float, torch.Tensor]]],
    datasets_list: list[list[str]],
):
    """merge metrics computed on different datasets.

    metrics_list[i] is a dictionary of metrics computed on datasets_list[i].
    datasets_list[i] is a list of datasets.

    Example:
        ```python
        metrics_list = [
            {"mae": 0.2, "ndcg_retrieve_sim": 0.9}, # computed on ["Grey1977"]
            {"mae": 0.3, "ndcg_retrieve_sim": 0.85} # computed on ["Patil2012_A3"]
        ]
        datasets_list = [
            ["Grey1977"],
            ["Patil2012_A3"]
        ]
        merged = merge_metrics(metrics_list, datasets_list)
        print(merged)  # {"mae": ..., "ndcg_retrieve_sim": ...} (weighted average)
        ```
    """
    assert len(metrics_list) == len(
        datasets_list
    ), "The number of metrics and datasets must match."
    first_metrics = set(metrics_list[0].keys())
    assert all(
        set(m.keys()) == first_metrics for m in metrics_list
    ), "Cannot merge metrics with different keys."
    seen_datasets = set()
    for ds in datasets_list:
        assert not (set(ds) & seen_datasets), "Duplicate datasets found."
        seen_datasets.update(ds)
    weights_list = [get_datasets_weights(ds) for ds in datasets_list]
    merged_metrics = {}
    for metric in metrics_list[0].keys():
        weights = [
            w[metric] if metric == "mae" else w["rank_based"] for w in weights_list
        ]
        merged_metrics[metric] = sum(
            [metrics[metric] * w for metrics, w in zip(metrics_list, weights)]
        ) / sum(weights)
    return merged_metrics
