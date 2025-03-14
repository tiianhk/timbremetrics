import os
import torch
import torch.nn as nn
from kymatio.torch import TimeFrequencyScattering

from timbremetrics.metrics import TimbreMetric
from timbremetrics.utils import print_results, write_results_to_yaml, AudioLoader
from timbremetrics.paths import BASE_DIR

out_file = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")
DEFAULT_SR = 44100


class jtfs(nn.Module):

    def __init__(
        self, J, Q, J_fr, Q_fr, T, F, keep_time_dimension=False, fixed_duration=2.0
    ):
        super().__init__()
        class_name = self.__class__.__name__
        self.name = class_name if keep_time_dimension else f"time-avg_{class_name}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.keep_time_dimension = keep_time_dimension
        # fix a duration since T has to be smaller than any input lengths
        self.fixed_duration = fixed_duration
        self.transform = TimeFrequencyScattering(
            shape=(int(self.fixed_duration * DEFAULT_SR),),
            J=J,
            Q=Q,
            J_fr=J_fr,
            Q_fr=Q_fr,
            T=T,
            F=F,
            format="joint",
        ).to(self.device)

    def forward(self, audio):
        y = self.transform(audio.to(self.device))
        if not self.keep_time_dimension:
            y = y.mean(dim=-1)
        return y.cpu()


model = jtfs(J=10, Q=(8, 2), J_fr=5, Q_fr=2, T=44100, F=2**4, keep_time_dimension=False)
metric = TimbreMetric(model, fixed_duration=2.0)
res = metric()
print_results(model.name, res)
write_results_to_yaml(out_file, model.name, res)

model = jtfs(J=10, Q=(8, 2), J_fr=5, Q_fr=2, T=44100, F=2**4, keep_time_dimension=True)
metric = TimbreMetric(model, fixed_duration=2.0)
res = metric()
print_results(model.name, res)
write_results_to_yaml(out_file, model.name, res)
