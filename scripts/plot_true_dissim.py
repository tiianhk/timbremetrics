from timbremetrics.utils import get_true_dissim, mask, min_max_normalization
from timbremetrics.paths import BASE_DIR
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
import os

true_dissim = get_true_dissim()
for d, mtx in true_dissim.items():
    true_dissim[d] = min_max_normalization(mask(mtx)).numpy()

idx1, idx2, idx3 = 0, 10, 20
dataset_names = list(true_dissim.keys())
matrices = list(true_dissim.values())
sparse_mtx = block_diag(matrices[idx1], matrices[idx2], matrices[idx3])
block_labels = [dataset_names[idx1], dataset_names[idx2], dataset_names[idx3]]
block_sizes = [
    matrices[idx1].shape[0],
    matrices[idx2].shape[0],
    matrices[idx3].shape[0],
]
starts = np.cumsum([0] + block_sizes[:-1])

rows, cols = sparse_mtx.nonzero()
values = sparse_mtx[rows, cols]
x = rows + cols
y = cols - rows

plt.figure(figsize=(6, 2))
scatter = plt.scatter(x, y, c=values, cmap="viridis", marker="s", s=5)

y_min = np.min(y)
label_y_position = y_min - 2  # Vertical position for labels

for label, start, size in zip(block_labels, starts, block_sizes):

    midpoint_x = 2 * start + (size - 1)

    plt.text(
        midpoint_x,
        label_y_position,
        label,
        ha="center",
        va="top",
        rotation=0,
        fontsize=8,
        color="black",
    )

cbar = plt.colorbar(scatter, pad=0.0, shrink=0.5, aspect=10)
cbar.outline.set_visible(False)
cbar.ax.set_yticks([0,1])
cbar.ax.set_yticklabels(["identical", "most dissimilar    "])
# cbar.ax.tick_params(axis='y', which='both', length=0)
cbar.ax.tick_params(labelsize=8, length=0)

plt.gca().set_aspect("equal")
plt.ylim(label_y_position - 2, np.max(y))
plt.axis("off")
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "../assets/true_dissim.png"),
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
