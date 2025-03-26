import os
import torch
import torch.nn as nn
from torchaudio.transforms import MFCC

from timbremetrics import TimbreMetric, print_results, write_results_to_yaml
from timbremetrics.paths import BASE_DIR

out_file = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")

DEFAULT_SR = 44100


class mfcc(nn.Module):
    def __init__(
        self,
        time_average=True,
        sample_rate=DEFAULT_SR,
        melkwargs={"n_fft": 2048, "hop_length": 512},
    ):
        super().__init__()
        class_name = self.__class__.__name__
        self.name = f"time-avg_{class_name}" if time_average else class_name
        self.time_average = time_average
        self.model = MFCC(sample_rate=sample_rate, melkwargs=melkwargs)

    def forward(self, x):
        x = self.model(x)
        if self.time_average:
            x = x.mean(dim=-1)
        return x


model = mfcc(time_average=True)
metric = TimbreMetric(pad_to_the_longer_length=False)
res = metric(model)
print_results(model.name, res)
write_results_to_yaml(out_file, model.name, res)

model = mfcc(time_average=False)
metric = TimbreMetric(pad_to_the_longer_length=True)
res = metric(model)
print_results(model.name, res)
write_results_to_yaml(out_file, model.name, res)
