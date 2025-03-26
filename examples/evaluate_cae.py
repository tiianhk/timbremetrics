import torch
import torch.nn as nn
import numpy as np

"""https://github.com/SonyCSLParis/cae-invar/blob/master/complex_auto/cqt.py"""


def standardize(x, axis=-1):
    """
    Performs contrast normalization (zero mean, unit variance)
    along the given axis.

    :param x: array to normalize
    :param axis: normalize along that axis
    :return: contrast-normalized array
    """
    stds_avg = np.std(x, axis=axis, keepdims=True)
    x -= np.mean(x, axis=axis, keepdims=True)
    x /= stds_avg + 1e-8
    return x


"""https://github.com/SonyCSLParis/cae-invar/blob/master/complex_auto/complex.py"""


class Complex(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.5, learn_norm=False):
        super(Complex, self).__init__()

        self.layer = nn.Linear(n_in, n_out * 2, bias=False)

        self.drop = nn.Dropout(dropout)
        self.learn_norm = learn_norm
        self.n_out = n_out
        self.norm_val = nn.Parameter(torch.Tensor([0.43]))  # any start value

    def drop_gauss(self, x):
        return torch.normal(mean=x, std=0.5)

    def forward(self, x):
        out = torch.matmul(
            self.drop(x), self.set_to_norm_graph(self.norm_val).transpose(0, 1)
        )
        real = out[:, : self.n_out]
        imag = out[:, self.n_out :]
        amplitudes = (real**2 + imag**2) ** 0.5
        phases = torch.atan2(real, imag)
        return amplitudes, phases

    def backward(self, amplitudes, phases):
        real = torch.sin(phases) * amplitudes
        imag = torch.cos(phases) * amplitudes
        cat_ = torch.cat((real, imag), dim=1)
        recon = torch.matmul(cat_, self.set_to_norm_graph(self.norm_val))
        return recon

    def set_to_norm(self, val):
        """
        Sets the norms of all convolutional kernels of the C-GAE to a specific
        value.

        :param val: norms of kernels are set to this value
        """
        if val == -1:
            val = self.norm_val
        shape_x = self.layer.weight.size()
        conv_x_reshape = self.layer.weight.view(shape_x[0], -1)
        norms_x = ((conv_x_reshape**2).sum(1) ** 0.5).view(-1, 1)
        conv_x_reshape = conv_x_reshape / norms_x
        weight_x_new = (conv_x_reshape.view(*shape_x) * val).clone()
        self.layer.weight.data = weight_x_new

    def set_to_norm_graph(self, val):
        if not self.learn_norm:
            return self.layer.weight
        """
        Sets the norms of all convolutional kernels of the C-GAE to a learned
        value.

        :param val: norms of kernels are set to this value
        """
        if val == -1:
            val = self.norm_val
        shape_x = self.layer.weight.size()
        conv_x_reshape = self.layer.weight.view(shape_x[0], -1)
        norms_x = ((conv_x_reshape**2).sum(1) ** 0.5).view(-1, 1)
        conv_x_reshape = conv_x_reshape / norms_x
        weight_x_new = (conv_x_reshape.view(*shape_x) * val).clone()
        return weight_x_new


"""
Code adapted from https://github.com/SonyCSLParis/cae-invar
Hyperparameters from https://github.com/SonyCSLParis/cae-invar/blob/master/config_cqt.ini
"""

import os
from functools import partial

import librosa
from timbremetrics import TimbreMetric, print_results, write_results_to_yaml
from timbremetrics.paths import BASE_DIR

out_file = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")
model_save_fn = os.path.join(
    os.path.dirname(BASE_DIR), "examples/model_complex_auto_cqt.save"
)


def _cae_compute(
    audio: torch.Tensor,
    model,
    sr=22050,
    n_bins=120,
    bins_per_oct=24,
    fmin=65.4,
    hop_length=1984,
    step_size=1,
):
    cqt = librosa.cqt(
        audio.squeeze(0).numpy(),
        sr=sr,
        n_bins=n_bins,
        bins_per_octave=bins_per_oct,
        fmin=fmin,
        hop_length=hop_length,
    )
    mag = librosa.magphase(cqt)[0]
    mag = standardize(mag, axis=0).transpose()
    ngrams = []
    for i in range(0, len(mag) - length_ngram, step_size):
        curr_ngram = mag[i : i + length_ngram].reshape((-1,))
        ngrams.append(curr_ngram)
    x = torch.tensor(np.vstack(ngrams))
    amp, phase = model(x)
    amp = amp[0]  # only the first frame (covers 2.88 seconds)
    return amp.detach()


n_bins = 120
length_ngram = 32
n_bases = 256
dropout = 0.5
model = Complex(n_bins * length_ngram, n_bases, dropout=dropout)
model.load_state_dict(torch.load(model_save_fn), strict=False)
model.eval()

cae_model = partial(_cae_compute, model=model)
metric = TimbreMetric(
    sample_rate=22050,
    pad_to_the_longer_length=False,
    fixed_duration=3.0,  # context window is 1984 (hop_length) * 32 (length_ngram) / 22050 (sample_rate) = 2.88
)
res = metric(cae_model)
print_results("cae", res)
write_results_to_yaml(out_file, "cae", res)
