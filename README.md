# Timbre Metrics

Measure how well your model’s audio embeddings match human timbre perception using 21 published datasets with audio files and their pairwise dissimilarity ratings.

Below, human dissimilarity ratings from three datasets are visualized as triangular matrices to exclude unordered duplicates and self-comparisons. For each pair of audio samples, a brighter color indicates that the two sounds are judged to be more dissimilar by human listeners.

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
