import os
import torch
import torch.nn as nn
from kymatio.torch import TimeFrequencyScattering

from timbremetrics import TimbreMetric, print_results, write_results_to_yaml
from timbremetrics.paths import BASE_DIR

out_file = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")


class jtfs(nn.Module):

    def __init__(self, J, Q, J_fr, Q_fr, T, F, time_average=True):
        super().__init__()
        class_name = self.__class__.__name__
        self.name = f"time-avg_{class_name}" if time_average else class_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.J = J
        self.Q = Q
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.T = T
        self.F = F
        self.time_average = time_average
        self.cnt = 0

    def forward(self, audio):
        self.cnt += 1
        print(f"jtfs: {self.cnt}")
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
        if self.time_average:
            y = y.mean(dim=-1)
        return y


model = jtfs(J=12, Q=(8, 2), J_fr=3, Q_fr=2, T=None, F=None, time_average=True)
metric = TimbreMetric(device=model.device)
res = metric(model)
print_results(model.name, res)
write_results_to_yaml(out_file, model.name, res)

# this will take some time!
model = jtfs(J=12, Q=(8, 2), J_fr=3, Q_fr=2, T=None, F=None, time_average=False)
metric = TimbreMetric(device=model.device, pad_to_the_longer_length=True)
res = metric(model)
print_results(model.name, res)
write_results_to_yaml(out_file, model.name, res)
