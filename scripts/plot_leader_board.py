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

# Convert to a dictionary of {metric: [model, config, score]}
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
sns.set_theme(style="darkgrid")

for metric, scores in metric_scores.items():

    model_groups = {}
    for model, config, score in scores:
        if model not in model_groups:
            model_groups[model] = []
        model_groups[model].append((config, score))

    model_avg_scores = {
        model: np.mean([s for _, s in data]) for model, data in model_groups.items()
    }

    # list of models sorted by average score
    sorted_models = sorted(model_avg_scores, key=model_avg_scores.get, reverse=True)

    # Organize data for plotting
    sorted_entries = []
    model_x_ranges = {}  # Store x positions for each model
    current_x = 0
    gap = 2  # Space between model groups

    for model in sorted_models:
        # Sort configs within the model by string name
        sorted_types = sorted(model_groups[model], key=lambda x: x[0])

        x_range = list(range(current_x, current_x + len(sorted_types)))
        model_x_ranges[model] = x_range
        sorted_entries.extend(
            [(c, s, x) for (c, s), x in zip(sorted_types, x_range)]
        )

        current_x += (
            len(sorted_types) + gap
        )  # Move x position for next model, adding a gap

    # Extract x-axis labels, scores, and x positions
    configs = [c for c, _, _ in sorted_entries]
    scores = [v for _, v, _ in sorted_entries]
    x_positions = [x for _, _, x in sorted_entries]

    fig, ax = plt.subplots(figsize=(30, 5))

    min_value = min(scores)
    max_value = max(scores)
    small_margin = 0.1 * (max_value - min_value)  # Small margin as 10% of the range

    colors = ["#0077BB", "#CC3311", "#009988", "#EE7733"]

    config_scores = {}

    # Plot each model separately, ensuring gaps between groups
    for model in sorted_models:
        x_range = model_x_ranges[model]
        y_values = [
            scores[x_positions.index(x)] for x in x_range
        ]  # Get scores for this model
        config_labels = [
            configs[x_positions.index(x)] for x in x_range
        ]

        ax.bar(
            x_range,
            y_values,
            color=colors[: len(x_range)],
            label=config_labels,
        )

        for i, c in enumerate(config_labels):
            if c not in config_scores:
                config_scores[c] = []
            config_scores[c].append((x_range[i], y_values[i]))

    best_tuple_entries = []
    for i, c in enumerate(config_scores):
        if metric == "mae":
            best_tuple = min(config_scores[c], key=lambda x: x[1])
        else:
            best_tuple = max(config_scores[c], key=lambda x: x[1])
        ax.scatter(
            best_tuple[0],
            best_tuple[1] + small_margin / 3,
            marker="*",
            color=colors[i],
            s=200,
            zorder=2,
        )
        best_tuple_entries.append(best_tuple)

    if metric == "mae":
        best_of_best = min(best_tuple_entries, key=lambda x: x[1])
    else:
        best_of_best = max(best_tuple_entries, key=lambda x: x[1])

    ax.scatter(
        best_of_best[0],
        best_of_best[1] + small_margin / 3,
        marker="o",
        color="gold",
        s=200,
        zorder=1,
    )

    model_centers = []
    # Add model labels under the grouped config labels
    for model, x_range in model_x_ranges.items():
        # Calculate the center of the x_range for the model
        model_centers.append(sum(x_range) / len(x_range))

    ax.set_xticks(model_centers)
    ax.set_xticklabels(list(model_x_ranges.keys()), rotation=15, fontsize=16)

    ax.set_ylabel(metric, fontsize=16)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)

    ax.set_ylim(min_value - small_margin, max_value + small_margin)

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # Use a dictionary to remove duplicates
    ax.legend(
        unique_labels.values(), unique_labels.keys(), loc="upper right", fontsize=16
    )

    plt.savefig(
        os.path.join(os.path.dirname(BASE_DIR), f"assets/{metric}_eval.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
