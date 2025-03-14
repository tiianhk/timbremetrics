import os
from music2latent import EncoderDecoder

from timbremetrics.metrics import TimbreMetric
from timbremetrics.utils import print_results, write_results_to_yaml
from timbremetrics.paths import BASE_DIR

out_file = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")

codec = EncoderDecoder()

# pad_to_max_duration can not satisfy the minimum duration requirement of the model
# so fixed_duration is used instead
metric = TimbreMetric(model=codec, fixed_duration=2.0)

m2l = lambda x: codec.encode(x, extract_features=True)
res = metric(m2l)
print_results('music2latent', res)
write_results_to_yaml(out_file, 'music2latent', res)

avg_m2l = lambda x: codec.encode(x, extract_features=True).mean(dim=-1)
res = metric(avg_m2l)
print_results('time-avg_music2latent', res)
write_results_to_yaml(out_file, 'time-avg_music2latent', res)
