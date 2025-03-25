import os
import torch
import torch.nn as nn
from kymatio.torch import TimeFrequencyScattering

from timbremetrics import TimbreMetric, print_results, write_results_to_yaml
from timbremetrics.paths import BASE_DIR

out_file = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")


class jtfs(nn.Module):

    def __init__(self, J, Q, J_fr, Q_fr, T, F, keep_time_dimension=False):
        super().__init__()
        class_name = self.__class__.__name__
        self.name = class_name if keep_time_dimension else f"time-avg_{class_name}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.J = J
        self.Q = Q
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.T = T
        self.F = F
        self.cnt = 0

    def forward(self, audio):
        self.cnt += 1
        print(f"jtfs: processing {self.cnt}-th sample")
        N = audio.shape[-1]
        transform = TimeFrequencyScattering(
            shape=(N,),
            J=self.J,
            Q=self.Q,
            J_fr=self.J_fr,
            Q_fr=self.Q_fr,
            T=self.T,
            F=self.F,
            format="time",
        ).to(self.device)
        y = transform(audio)
        return y


model = jtfs(
    J=12, Q=(8, 2), J_fr=3, Q_fr=2, T="global", F=None, keep_time_dimension=False
)
metric = TimbreMetric(device=model.device)
res = metric(model)
print_results(model.name, res)
write_results_to_yaml(out_file, model.name, res)

model = jtfs(J=12, Q=(8, 2), J_fr=3, Q_fr=2, T=None, F=None, keep_time_dimension=True)
metric = TimbreMetric(device=model.device, pad_to_max_duration=True)
res = metric(model)
print_results(model.name, res)
write_results_to_yaml(out_file, model.name, res)
