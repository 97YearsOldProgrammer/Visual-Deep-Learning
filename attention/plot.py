"""Attention heatmap plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_attention(
    attention: npt.NDArray[np.floating],
    *,
    tokens: list[str] | None = None,
    cmap: str = "Blues",
    title: str | None = None,
    xlabel: str = "Key",
    ylabel: str = "Query",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a single attention heatmap.

    Args:
        attention: (S, S) attention matrix.
        tokens: Optional token labels for axes.
        cmap: Matplotlib colormap name.
        title: Optional figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        ax: Optional existing Axes.

    Returns:
        (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    im = ax.imshow(attention, cmap=cmap, aspect="auto", vmin=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if tokens is not None:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return fig, ax


def plot_attention_grid(
    attention_heads: npt.NDArray[np.floating],
    *,
    tokens: list[str] | None = None,
    cmap: str = "Blues",
    ncols: int = 4,
    head_labels: list[str] | None = None,
) -> tuple[plt.Figure, npt.NDArray]:
    """Plot a grid of attention heads.

    Args:
        attention_heads: (H, S, S) multi-head attention matrix.
        tokens: Optional token labels.
        cmap: Matplotlib colormap.
        ncols: Number of columns in the grid.
        head_labels: Optional labels for each head.

    Returns:
        (fig, axes) tuple where axes is a 2-D array.
    """
    n_heads = attention_heads.shape[0]
    nrows = (n_heads + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = np.atleast_2d(axes)

    for i in range(n_heads):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        ax.imshow(attention_heads[i], cmap=cmap, aspect="auto", vmin=0)

        label = head_labels[i] if head_labels else f"Head {i}"
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for i in range(n_heads, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    return fig, axes
