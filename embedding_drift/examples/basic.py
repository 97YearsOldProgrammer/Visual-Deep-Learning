"""Example: embedding drift visualization with synthetic data."""

import numpy as np

from lib.theme import apply_theme
from embedding_drift.plot import plot_drift

apply_theme()

rng = np.random.default_rng(42)
n = 60

# "Before" embeddings: two clusters
before = np.vstack([
    rng.normal([0, 0], 0.5, (n // 2, 2)),
    rng.normal([3, 3], 0.5, (n // 2, 2)),
])

# "After" embeddings: clusters move apart + some rotation
after = np.vstack([
    rng.normal([-1, -1], 0.4, (n // 2, 2)),
    rng.normal([4, 5], 0.4, (n // 2, 2)),
])

labels = np.array([0] * (n // 2) + [1] * (n // 2))

fig, ax = plot_drift(
    before, after,
    labels=labels,
    label_names=["Class A", "Class B"],
    title="Embedding Drift",
)
fig.savefig("embedding_drift_example.pdf")
print("Saved embedding_drift_example.pdf")
