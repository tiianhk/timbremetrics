import torch
import torch.nn as nn
from torchaudio.transforms import MFCC

from timbremetrics.metrics import TimbreMetric
from timbremetrics.utils import print_results

DEFAULT_SR = 44100


class mfcc(nn.Module):
    def __init__(
        self,
        keep_time_dimension=False,
        sample_rate=DEFAULT_SR,
        melkwargs={"n_fft": 2048, "hop_length": 512},
    ):
        super().__init__()
        self.name = ", ".join(
            [
                f"{self.__class__.__name__}",
                f"keep_time_dimension={keep_time_dimension}",
                f"sample_rate={sample_rate}",
                f"melkwargs={melkwargs}",
            ]
        )
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
model = mfcc(keep_time_dimension=True)
metric = TimbreMetric(model, sample_rate=model.sr, pad_to_max_duration=True)
res = metric()
print_results(model.name, res)
