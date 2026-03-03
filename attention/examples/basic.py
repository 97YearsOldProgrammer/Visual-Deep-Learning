"""Example: attention heatmap with synthetic data."""

import numpy as np
from scipy.special import softmax

from lib.theme import apply_theme
from attention.plot import plot_attention, plot_attention_grid
from attention.rollout import attention_rollout

apply_theme()

rng = np.random.default_rng(42)
seq_len = 8
n_heads = 6
n_layers = 4
tokens = ["[CLS]", "The", "cat", "sat", "on", "the", "mat", "[SEP]"]

# Synthetic multi-head attention across layers
layers = []
for _ in range(n_layers):
    raw = rng.normal(0, 1, (n_heads, seq_len, seq_len))
    attn = softmax(raw, axis=-1)
    layers.append(attn)

# Single head heatmap
fig, ax = plot_attention(layers[0][0], tokens=tokens, title="Layer 0, Head 0")
fig.savefig("attention_single.pdf")
print("Saved attention_single.pdf")

# Multi-head grid
fig, axes = plot_attention_grid(layers[0], tokens=tokens)
fig.savefig("attention_grid.pdf")
print("Saved attention_grid.pdf")

# Attention rollout
rollout = attention_rollout(layers)
fig, ax = plot_attention(rollout, tokens=tokens, title="Attention Rollout", cmap="Reds")
fig.savefig("attention_rollout.pdf")
print("Saved attention_rollout.pdf")
