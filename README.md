# Timbre Metrics

Measure how well your model’s audio embeddings match human timbre perception using 21 published datasets with audio files and their pairwise (dis)similarity ratings.

Below, human (dis)similarity ratings from three datasets are visualized as triangular matrices. Each point corresponds to a unique audio pair. A darker color indicates that the sounds are judged to be more similar by human listeners.

![Dissimilarities between audio stimuli judged by humans](assets/true_dissim.png)

These values represent perceptual timbre similarity, as in nearly all cases, paired sounds are controlled to have the same pitch, loudness, and duration.

## Installation
```
pip install git+https://github.com/tiianhk/timbremetrics.git
```
To install as an editable package, first git clone this repository and open its directory, then say
```
pip install -e .
```
To install with extra dependencies to evaluate pre-trained models, DSP models, and to visualize, say
```
pip install -e .[extra]
```
To evaluate models loaded by [fadtk](https://github.com/microsoft/fadtk), Python version 3.11.6 is recommanded.

## Usage
```
from timbremetrics.metrics import TimbreMetric

metric = TimbreMetric(model)

evaluation_results = metric()
```
