# Timbre Metrics

Measure how well your model’s audio embeddings match human timbre perception using 21 published datasets with audio files and pairwise dissimilarity ratings (color-coded in the figure below).

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
