import yaml
import matplotlib.pyplot as plt
import numpy as np
import os

from timbremetrics.paths import BASE_DIR

# Load YAML data from a pseudo file path
file_path = os.path.join(os.path.dirname(BASE_DIR), "examples/results.yaml")

with open(file_path, "r") as file:
    data = yaml.safe_load(file)

# Flatten the data into a list of (model, distance, metric, score)
entries = []
for model, distances in data.items():
    for distance, metrics in distances.items():
        for metric, score in metrics.items():
            if score is not None and not np.isnan(score):
                entries.append((model, distance, metric, score))

# Convert to a dictionary of {metric: [(model-distance, score)]}
metric_scores = {}
for model, distance, metric, score in entries:
    key = f"{model}-{distance}"  # Unique identifier
    if metric not in metric_scores:
        metric_scores[metric] = []
    metric_scores[metric].append((key, score))

# Define ranking order: MAE (lower is better), others (higher is better)
ranking_order = {
    "mae": False,  # Lower is better
    "kendall_corr": True,
    "ndcg_retrieve_sim": True,
    "spearman_corr": True,
    "triplet_agreement": True,
}

# Plot separate figures for each metric
for metric, scores in metric_scores.items():
    # Sort scores based on ranking order
    scores.sort(key=lambda x: x[1], reverse=ranking_order[metric])

    # Extract labels and values
    labels, values = zip(*scores)

    # Plot
    plt.figure(figsize=(40, 10))
    plt.bar(labels, values, color="skyblue")
    plt.xticks(rotation=90)  # Rotate labels for readability
    plt.ylabel("Score")
    plt.title(metric.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(f"{metric}.png", dpi=300)
