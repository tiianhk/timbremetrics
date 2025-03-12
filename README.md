# Timbre Metrics

Measure how well your model’s audio embeddings match human timbre perception using 21 published datasets with audio files and their pairwise dissimilarity ratings.

Below, human dissimilarity ratings from three datasets are visualized as triangular matrices to exclude unordered duplicates and self-comparisons (for $n$ sounds, there are $n\times(n-1)/2$ such pairs). For each pair, a brighter color indicates that the two sounds are judged to be more dissimilar by human listeners. This reflects the perceptual dissimilarity of timbre, since (almost every) in-pair sounds are controlled to have the same pitch, loudness, and duration.

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
