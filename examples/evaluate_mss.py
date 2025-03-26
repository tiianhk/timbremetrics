import os
import torch
import torch.nn as nn

from timbremetrics import TimbreMetric, print_results, write_results_to_yaml
from timbremetrics.paths import BASE_DIR

out_file = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")


class mss(nn.Module):
    """DDSP https://arxiv.org/abs/2001.04643v1"""

    def __init__(self, time_average=True, fft_sizes=(4096, 2048, 1024, 512, 256, 128)):
        super().__init__()
        class_name = self.__class__.__name__
        self.name = f"time-avg_{class_name}" if time_average else class_name
        self.time_average = time_average
        self.fft_sizes = fft_sizes

    def forward(self, audio):
        specs, log_specs = [], []
        for fft_size in self.fft_sizes:
            stft = torch.stft(audio, n_fft=fft_size, return_complex=True)
            mag_spec = torch.abs(stft)  # Shape: (batch, freq_bins, time_frames)
            safe_mag_spec = torch.where(
                mag_spec <= 1e-5, torch.tensor(1e-5, device=mag_spec.device), mag_spec
            )  # Avoid log(0)
            log_spec = torch.log(safe_mag_spec)
            if self.time_average:
                mag_spec = mag_spec.mean(dim=-1)
                log_spec = log_spec.mean(dim=-1)
            specs.append(mag_spec.flatten(start_dim=1))
            log_specs.append(log_spec.flatten(start_dim=1))
        specs = torch.cat(specs, dim=1)
        log_specs = torch.cat(log_specs, dim=1)
        return torch.cat([specs, log_specs], dim=1)


model = mss(time_average=True)
metric = TimbreMetric(pad_to_the_longer_length=False)
res = metric(model)
print_results(model.name, res)
write_results_to_yaml(out_file, model.name, res)

model = mss(time_average=False)
metric = TimbreMetric(pad_to_the_longer_length=True)
res = metric(model)
print_results(model.name, res)
write_results_to_yaml(out_file, model.name, res)
