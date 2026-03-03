"""Example: t-SNE visualization of synthetic clusters."""

import numpy as np

from lib.theme import apply_theme
from tsne.plot import plot_tsne

apply_theme()

rng = np.random.default_rng(42)
n_per_class = 100
dim = 64

# Three Gaussian clusters in high-D space
centers = rng.normal(0, 5, (3, dim))
embeddings = np.vstack([centers[i] + rng.normal(0, 1, (n_per_class, dim)) for i in range(3)])
labels = np.repeat([0, 1, 2], n_per_class)

fig, ax = plot_tsne(
    embeddings,
    labels,
    label_names=["Class A", "Class B", "Class C"],
    title="t-SNE Embedding",
)
fig.savefig("tsne_example.pdf")
print("Saved tsne_example.pdf")
