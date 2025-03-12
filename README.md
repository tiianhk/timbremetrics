# Timbre Metrics

Evaluate how closely your audio model aligns with human timbre perception by measuring the difference between the predicted dissimilarity matrix (pairwise distances of audio embeddings) and the true dissimilarity matrix (human ratings). This repository contains 21 small datasets collected by past timbre space studies. The figure below demonstrates three of them, where color represents the dissimilarity level between audio stimulus pairs.

![Dissimilarities between audio stimuli judged by humans](assets/true_dissim.png)

## Installation
```
pip install git+https://github.com/tiianhk/timbremetrics.git
```

## Usage
```
from timbremetrics.metrics import TimbreMetric

metric = TimbreMetric(model)

evaluation_results = metric()
```
