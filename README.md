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
If you want to run python scripts in the `examples/` folder to evaluate pre-trained models like CLAP and DSP features such as MFCCs, use `[extra]` to include extra dependencies:
```
pip install -e .[extra]
```
Especially, Python version 3.11.6 is recommanded if loading pre-trained models with [fadtk](https://github.com/microsoft/fadtk).

## Usage
to be finished
