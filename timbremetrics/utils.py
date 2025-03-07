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


def load_audio(
    dataset, audio_file, device=None, dtype=None, target_sr=None, fixed_duration=None
):

    f = os.path.join(STIMULI_DIR, dataset, audio_file)
    audio, sr = torchaudio.load(f, backend="soundfile")

    audio = audio.to(device=device, dtype=dtype)
    if target_sr is not None and sr != target_sr:
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
        sr = target_sr
    if fixed_duration is not None:
        target_num_samples = int(fixed_duration * sr)
        if audio.shape[-1] > target_num_samples:
            audio = audio[..., :target_num_samples]
        elif audio.shape[-1] < target_num_samples:
            padding = target_num_samples - audio.shape[-1]
            audio = F.pad(audio, (0, padding))

    return audio, sr


def load_dataset_audio(
    dataset, device=None, dtype=None, target_sr=None, fixed_duration=None
):

    audio_files = os.listdir(os.path.join(STIMULI_DIR, dataset))
    audio_files = sorted(audio_files)

    audio_data = []
    for audio_file in audio_files:
        if os.path.splitext(audio_file)[1] != ".aiff":
            continue

        audio, sr = load_audio(
            dataset,
            audio_file,
            device=device,
            dtype=dtype,
            target_sr=target_sr,
            fixed_duration=fixed_duration,
        )
        audio_data.append({"file": audio_file, "audio": audio, "sample_rate": sr})

    return audio_data


def get_audio(device=None, dtype=None, target_sr=None, fixed_duration=None):
    datasets = list_datasets()
    dataset_audio = {}
    for d in datasets:
        dataset_audio[d] = load_dataset_audio(
            d,
            device=device,
            dtype=dtype,
            target_sr=target_sr,
            fixed_duration=fixed_duration,
        )
    return dataset_audio


def load_dissimilarity_matrix(dataset, device=None):
    f = os.path.join(TRUE_DISSIM_DIR, f"{dataset}_dissimilarity_matrix.txt")
    return torch.tensor(np.loadtxt(f)).to(device)


def get_true_dissim(device=None):
    datasets = list_datasets()
    true_dissim = {}
    for d in datasets:
        data = load_dissimilarity_matrix(d, device=device)
        true_dissim[d] = min_max_normalization(mask(data))
    return true_dissim
