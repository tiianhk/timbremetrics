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
        fadtk_model=None,
        device=None,
        dtype=None,
        target_sr=None,
        pad_to_max_duration=False,
        fixed_duration=None,
    ):
        self.fadtk_model = fadtk_model
        self.device = device
        self.dtype = dtype
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
                    padding = max_sample_num - x["audio"].shape[-1]
                    if isinstance(x["audio"], torch.Tensor):
                        x["audio"] = F.pad(x["audio"], (0, padding))
                    elif isinstance(x["audio"], np.ndarray):
                        assert self.fadtk_model is not None
                        pad_width = [(0, 0)] * (x["audio"].ndim - 1) + [(0, padding)]
                        x["audio"] = np.pad(x["audio"], pad_width)
                    else:
                        assert self.fadtk_model is not None
                        # for descript audio codec
                        from audiotools import AudioSignal

                        assert isinstance(x["audio"], AudioSignal)
                        assert isinstance(x["audio"].audio_data, torch.Tensor)
                        x["audio"].audio_data = F.pad(
                            x["audio"].audio_data, (0, padding)
                        )
        return audio_dataset

    def _load_one_audio_file(self, dataset, audio_file):
        f = os.path.join(STIMULI_DIR, dataset, audio_file)
        if self.fadtk_model is not None:
            audio = self.fadtk_model.load_wav(f)
            sr = self.fadtk_model.sr
        else:
            audio, sr = torchaudio.load(f, backend="soundfile")
            audio = audio.to(device=self.device, dtype=self.dtype)
            if self.target_sr is not None and sr != self.target_sr:
                audio = torchaudio.transforms.Resample(sr, self.target_sr)(audio)
                sr = self.target_sr
            if self.fixed_duration is not None:
                assert self.pad_to_max_duration is False
                target_sample_num = int(self.fixed_duration * sr)
                if audio.shape[-1] > target_sample_num:
                    audio = audio[..., :target_sample_num]
                elif audio.shape[-1] < target_sample_num:
                    padding = target_sample_num - audio.shape[-1]
                    audio = F.pad(audio, (0, padding))
        return audio, sr


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
            print(f"        {metric}: {value.item():.4f}")
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
