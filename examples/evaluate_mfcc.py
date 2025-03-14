import os
import torch
import torch.nn as nn
from torchaudio.transforms import MFCC

from timbremetrics.metrics import TimbreMetric
from timbremetrics.utils import print_results, write_results_to_yaml
from timbremetrics.paths import BASE_DIR

out_file = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")

DEFAULT_SR = 44100


class mfcc(nn.Module):
    def __init__(
        self,
        keep_time_dimension=False,
        sample_rate=DEFAULT_SR,
        melkwargs={"n_fft": 2048, "hop_length": 512},
    ):
        super().__init__()
        class_name = self.__class__.__name__
        self.name = class_name if keep_time_dimension else f"time-avg_{class_name}"
        self.keep_time_dimension = keep_time_dimension
        self.sr = sample_rate
        self.melkwargs = melkwargs
        self.model = MFCC(sample_rate=sample_rate, melkwargs=melkwargs)

    def forward(self, x):
        x = self.model(x)
        if not self.keep_time_dimension:
            x = x.mean(dim=-1)
        return x


model = mfcc()
metric = TimbreMetric(model, sample_rate=model.sr)
res = metric()
print_results(model.name, res)
write_results_to_yaml(out_file, model.name, res)

model = mfcc(keep_time_dimension=True)
metric = TimbreMetric(model, sample_rate=model.sr, pad_to_max_duration=True)
res = metric()
print_results(model.name, res)
write_results_to_yaml(out_file, model.name, res)
