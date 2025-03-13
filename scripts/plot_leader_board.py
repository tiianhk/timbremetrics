import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from timbremetrics.paths import BASE_DIR


file_path = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")

with open(file_path, "r") as file:
    data = yaml.safe_load(file)

# Flatten the data into a list of (model, distance, metric, score)
entries = []
for model, distances in data.items():
    for distance, metrics in distances.items():
        for metric, score in metrics.items():
            entries.append((model, distance, metric, score))

# Convert to a dictionary of {metric: [(model-distance, score)]}
metric_scores = {}
for model, distance, metric, score in entries:
    key = f"{model}-{distance}"  # Unique identifier
    if metric not in metric_scores:
        metric_scores[metric] = []
    metric_scores[metric].append((key, score))

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "serif"

for metric, scores in metric_scores.items():
    # Sort descendingly
    sorted_entries = sorted(scores, key=lambda x: x[1], reverse=True)
    keys = [k for k, _ in sorted_entries]
    scores = [v for _, v in sorted_entries]
    x = range(len(keys))

    fig, ax = plt.subplots(figsize=(30, 10))
    ax.plot(x, scores, "o-", markersize=4, linewidth=1, color="blue")

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=90)
    ax.set_ylabel(metric)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(
        os.path.join(os.path.dirname(BASE_DIR), f"assets/{metric}_eval.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
