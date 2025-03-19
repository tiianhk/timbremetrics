# Timbre Metrics

Measure how well your model’s audio embeddings match human timbre perception using 21 published datasets with audio files and their pairwise (dis)similarity ratings.

Below, human (dis)similarity ratings from three datasets are visualized as triangular matrices. Each point corresponds to a unique audio pair. A darker color indicates that the sounds are judged to be more similar by human listeners.

![Dissimilarities between audio stimuli judged by humans](assets/true_dissim.png)

These values represent perceptual timbre similarity, as in nearly all cases, paired sounds are controlled to have the same pitch, loudness, and duration.

## Installation
Run
```
pip install git+https://github.com/tiianhk/timbremetrics.git
```
or, to install as an editable package
```
git clone https://github.com/tiianhk/timbremetrics.git
cd timbremetrics
pip install -e .
```
If you want to run Python scripts in the `examples/` folder to evaluate pre-trained neural networks like CLAP and signal processing based features such as the joint time-frequency scattering transform (jTFS), use `[extra]` to install additional dependencies:
```
pip install -e .[extra]
```
Python version 3.11.6 is recommanded if using pre-trained models with [fadtk](https://github.com/microsoft/fadtk).

## Usage
A minimal example
```
from timbremetrics import TimbreMetric, print_results

# initialization and data loading
metric = TimbreMetric()

# compute the metrics
results = metric(model)

# print the results
print_results(model_name, results)
```
The printed results for `model` that computes the MFCC (see `examples/evaluate_mfcc.py`)
```
mfcc:
    l2:
        mae: 0.168
        ndcg_retrieve_sim: 0.905
        spearman_corr: 0.442
        kendall_corr: 0.342
        triplet_agreement: 0.722
    cosine:
        mae: 0.281
        ndcg_retrieve_sim: 0.907
        spearman_corr: 0.451
        kendall_corr: 0.351
        triplet_agreement: 0.726
```
Some options
```
# load data to gpu
metric = TimbreMetric(device='cuda')

# the default sample rate is 44100 Hz, you can change it to suit your model
metric = TimbreMetric(sample_rate=48000)

# audio lengths are different, you can pad them to the maximum length in one dataset
metric = TimbreMetric(pad_to_max_duration=True)

# you can pad or truncate all audio to a fixed duration
metric = TimbreMetric(fixed_duration=2.0)
```
The `model` should be a `Callable` (e.g., functions, methods, lambdas, or objects that implement the `__call__` method) and should produce embeddings for audio tensors of shape `(1, num_samples)`.
The output tensors must have the same shape to allow for pairwise distance computation.

If used during model training, it is recommended to initialize the object before training starts and use it to compute metrics once per validation epoch (e.g., in `on_validation_epoch_end()` of a Lightning module).

See [here](timbremetrics/metrics.py#L86-L107) for more options to initialize an object using the `TimbreMetric` class. For practical examples, check the files in `examples/`.

## Acknowledgement
Data source:
 - https://github.com/EtienneTho/musical-timbre-studies
 - https://github.com/ben-hayes/timbre-dissimilarity-metrics

Code:
 - Much of this work has been adapted from https://github.com/ben-hayes/timbre-dissimilarity-metrics
 - Some code was taken without modification from https://github.com/leymir/hyperbolic-image-embeddings

Thanks to the authors for their work!
