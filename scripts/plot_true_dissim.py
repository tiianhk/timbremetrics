from timbremetrics.utils import get_true_dissim, mask, min_max_normalization
from timbremetrics.paths import BASE_DIR
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
import os

true_dissim = get_true_dissim()
for d, mtx in true_dissim.items():
    mtx = min_max_normalization(mask(mtx)).numpy()
    true_dissim[d] = mtx + mtx.T

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
alphas = np.where(rows < cols, 1.0, 0.2)

plt.figure(figsize=(6, 6))
scatter = plt.scatter(
    cols, rows, c=values, alpha=alphas, cmap="YlGnBu", marker="s", s=5
)

label_x = starts[1] + 1
label_y = starts[0] + block_sizes[0] // 2
plt.text(
    label_x,
    label_y,
    f"\u2192 {block_labels[0]}",
    ha="left",
    va="center",
    rotation=0,
    fontsize=12,
    color="black",
)

label_x = starts[2] + 1
label_y = starts[1] + block_sizes[1] // 2
plt.text(
    label_x,
    label_y,
    f"\u2192 {block_labels[1]}",
    ha="left",
    va="center",
    rotation=0,
    fontsize=12,
    color="black",
)

label_x = starts[2] - 2
label_y = starts[2] + block_sizes[2] // 2
plt.text(
    label_x,
    label_y,
    f"{block_labels[2]} \u2190",
    ha="right",
    va="center",
    rotation=0,
    fontsize=12,
    color="black",
)

cbar = plt.colorbar(scatter, pad=0.0, shrink=0.7, aspect=20, orientation="horizontal")
cbar.outline.set_visible(False)
cbar.ax.set_xticks([0, 1])
cbar.ax.set_xticklabels(["identical", "most dissimilar"])
cbar.ax.tick_params(labelsize=12, length=0)

plt.gca().set_aspect("equal")
plt.gca().invert_yaxis()
plt.axis("off")
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_DIR, "../assets/true_dissim.png"),
    bbox_inches="tight",
    dpi=300,
)
