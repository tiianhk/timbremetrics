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
        device=None,
        dtype=None,
        fadtk_audio_loader=None,
        target_sr=None,
        pad_to_max_duration=False,
        fixed_duration=None,
    ):
        self.device = device
        self.dtype = dtype
        self.fadtk_audio_loader = fadtk_audio_loader
        self.target_sr = target_sr
        self.pad_to_max_duration = pad_to_max_duration
        self.fixed_duration = fixed_duration

    def _load_audio_datasets(self):
        datasets = list_datasets()
        audio_datasets = {}
        for d in datasets:
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
        if self.pad_to_max_duration:
            assert self.fixed_duration is None
            max_sample_num = max([x["audio"].shape[-1] for x in audio_dataset])
            for x in audio_dataset:
                x["audio"] = self._to_target_length(x["audio"], max_sample_num)
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
            assert self.pad_to_max_duration is False
            target_sample_num = int(self.fixed_duration * sr)
            audio = self._to_target_length(audio, target_sample_num)
        return audio, sr

    def _to_target_length(self, audio, target_sample_num: int):
        if isinstance(audio, torch.Tensor):
            if audio.shape[-1] > target_sample_num:
                audio = audio[..., :target_sample_num]
            elif audio.shape[-1] < target_sample_num:
                padding = target_sample_num - audio.shape[-1]
                audio = F.pad(audio, (0, padding))
        elif isinstance(audio, np.ndarray):
            assert self.fadtk_audio_loader is not None
            if audio.shape[-1] > target_sample_num:
                audio = audio[..., :target_sample_num]
            elif audio.shape[-1] < target_sample_num:
                padding = target_sample_num - audio.shape[-1]
                pad_width = [(0, 0)] * (audio.ndim - 1) + [(0, padding)]
                audio = np.pad(audio, pad_width)
        else:
            assert self.fadtk_audio_loader is not None
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


def get_true_dissim(device=None):
    datasets = list_datasets()
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
