"""Example: UMAP visualization of synthetic clusters."""

import numpy as np

from lib.theme import apply_theme
from umap_viz.plot import plot_umap

apply_theme()

rng = np.random.default_rng(42)
n_per_class = 150
dim = 128

centers = rng.normal(0, 3, (4, dim))
embeddings = np.vstack([centers[i] + rng.normal(0, 1, (n_per_class, dim)) for i in range(4)])
labels = np.repeat([0, 1, 2, 3], n_per_class)

fig, ax = plot_umap(
    embeddings,
    labels,
    label_names=["Cluster A", "Cluster B", "Cluster C", "Cluster D"],
    title="UMAP Projection",
)
fig.savefig("umap_example.pdf")
print("Saved umap_example.pdf")
