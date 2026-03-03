"""Example: plot a synthetic training loss curve."""

import numpy as np

from lib.theme import apply_theme
from loss_curve.plot import plot_loss

apply_theme()

rng = np.random.default_rng(42)
steps = 500
train_loss = 2.0 * np.exp(-np.linspace(0, 4, steps)) + rng.normal(0, 0.05, steps)
val_loss = 2.0 * np.exp(-np.linspace(0, 4, steps // 5)) + rng.normal(0, 0.03, steps // 5) + 0.1

fig, ax = plot_loss(train_loss, val_loss, title="Training Loss")
fig.savefig("loss_curve_example.pdf")
print("Saved loss_curve_example.pdf")
