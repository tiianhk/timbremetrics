# Timbre Metrics

Measure how well your model’s audio embeddings match human timbre perception using 21 published datasets with audio files and their pairwise (dis)similarity ratings.

Below, human (dis)similarity ratings from three datasets are visualized as triangular matrices to exclude unordered duplicates and self-comparisons (for $n$ sounds, there are $n\times(n-1)/2$ such pairs). For each pair, a darker color indicates that the two sounds are judged to be more similar by human listeners. This reflects the perceptual similarity of timbre, since (almost every) paired sounds are controlled to have the same pitch, loudness, and duration.

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
