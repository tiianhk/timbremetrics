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

# Convert to a dictionary of {metric: [(model_distance, score)]}
metric_scores = {}
for model, distance, metric, score in entries:
    if metric not in metric_scores:
        metric_scores[metric] = []
    splits = model.split("_")
    if len(splits) == 1:
        metric_scores[metric].append((model, distance, score))
    else:
        assert len(splits) == 2
        metric_scores[metric].append((splits[1], f"{splits[0]}_{distance}", score))

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "serif"

for metric, scores in metric_scores.items():

    model_groups = {}
    for model, config, score in scores:
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append((config, score))

    model_avg_scores = {
        model: np.mean([s for _, s in data]) for model, data in model_groups.items()
    }

    sorted_models = sorted(model_avg_scores, key=model_avg_scores.get, reverse=True)

    # Organize data for plotting
    sorted_entries = []
    model_x_ranges = {}  # Store x positions for each model
    current_x = 0
    gap = 2  # Space between model groups

    for model in sorted_models:
        # Sort the 8 types within the model by score (high to low)
        sorted_types = sorted(model_groups[model], key=lambda x: x[1], reverse=True)

        x_range = list(range(current_x, current_x + len(sorted_types)))
        model_x_ranges[model] = x_range
        sorted_entries.extend(
            [(f"{model}_{d}", s, x) for (d, s), x in zip(sorted_types, x_range)]
        )

        current_x += (
            len(sorted_types) + gap
        )  # Move x position for next model, adding a gap

    # Extract x-axis labels, scores, and x positions
    keys = [k for k, _, _ in sorted_entries]
    scores = [v for _, v, _ in sorted_entries]
    x_positions = [x for _, _, x in sorted_entries]

    fig, ax = plt.subplots(figsize=(30, 10))

    # Generate distinct colors
    cmap = plt.colormaps["tab10"]  # Get a colormap (10 distinct colors)
    colors = {model: cmap(i % 10) for i, model in enumerate(sorted_models)}

    # Plot each model separately, ensuring gaps between groups
    for model in sorted_models:
        x_range = model_x_ranges[model]
        y_values = [
            scores[x_positions.index(x)] for x in x_range
        ]  # Get scores for this model

        ax.plot(
            x_range,
            y_values,
            "o-",
            markersize=4,
            linewidth=2,
            color=colors[model],
            label=model,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(keys, rotation=90)
    ax.set_ylabel(metric)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)

    # Add legend for models
    ax.legend(title="Models", loc="upper right")

    plt.savefig(
        os.path.join(os.path.dirname(BASE_DIR), f"assets/{metric}_eval.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
