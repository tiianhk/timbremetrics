from timbremetrics.utils import get_true_dissim, mask, min_max_normalization
from timbremetrics.paths import BASE_DIR
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
import os

true_dissim = get_true_dissim()
for d, mtx in true_dissim.items():
    true_dissim[d] = min_max_normalization(mask(mtx)).numpy()

sparse_mtx = block_diag(*list(true_dissim.values()))
rows, cols = sparse_mtx.nonzero()
values = sparse_mtx[rows, cols]
x = rows + cols
y = cols - rows

block_labels = list(true_dissim.keys())
block_sizes = [m.shape[0] for m in true_dissim.values()]
starts = np.cumsum([0] + block_sizes[:-1])

plt.figure(figsize=(12, 3))
scatter = plt.scatter(x, y, c=values, cmap="viridis", marker="s", s=1)

y_min = np.min(y)
label_y_position = y_min - 5  # Vertical position for labels

for label, start, size in zip(block_labels, starts, block_sizes):

    midpoint_x = 2 * start + (size - 1)

    plt.text(
        midpoint_x,
        label_y_position,
        label,
        ha="center",
        va="top",
        rotation=90,
        fontsize=8,
        color="black",
    )

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
