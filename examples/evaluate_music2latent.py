import os
import torch.nn.functional as F
from music2latent import EncoderDecoder

from timbremetrics import TimbreMetric, print_results, write_results_to_yaml
from timbremetrics.paths import BASE_DIR

out_file = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")


def pad(x):
    if x.shape[-1] < 5632:
        x = F.pad(x, (0, 5632 - x.shape[-1]))
    return x


codec = EncoderDecoder()

model = lambda x: codec.encode(pad(x), extract_features=True).mean(dim=-1)
metric = TimbreMetric(device=codec.device, pad_to_the_longer_length=False)
res = metric(model)
print_results("time-avg_music2latent", res)
write_results_to_yaml(out_file, "time-avg_music2latent", res)

model = lambda x: codec.encode(pad(x), extract_features=True)
metric = TimbreMetric(device=codec.device, pad_to_the_longer_length=True)
res = metric(model)
print_results("music2latent", res)
write_results_to_yaml(out_file, "music2latent", res)
