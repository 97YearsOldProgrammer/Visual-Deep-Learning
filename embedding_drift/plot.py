"""Embedding drift visualization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from embedding_drift.align import procrustes_align
from lib.theme import PALETTE


def plot_drift(
    before: npt.NDArray[np.floating],
    after: npt.NDArray[np.floating],
    *,
    labels: npt.NDArray[np.integer] | None = None,
    label_names: list[str] | None = None,
    align: bool = True,
    arrow_alpha: float = 0.3,
    point_size: float = 12,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot before/after embedding comparison with drift arrows.

    Both inputs should be 2-D arrays (N, 2), e.g. after running
    t-SNE or UMAP on the original high-dimensional embeddings.

    Args:
        before: (N, 2) coordinates before training/finetuning.
        after: (N, 2) coordinates after training/finetuning.
        labels: Optional (N,) integer class labels for coloring.
        label_names: Optional names for each class.
        align: Whether to apply Procrustes alignment first.
        arrow_alpha: Alpha for drift arrows.
        point_size: Scatter point size.
        title: Optional figure title.
        ax: Optional existing Axes.

    Returns:
        (fig, ax) tuple.
    """
    if align:
        before, after, _ = procrustes_align(before, after)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if labels is None:
        # Draw arrows from before to after
        for i in range(len(before)):
            ax.annotate(
                "",
                xy=after[i],
                xytext=before[i],
                arrowprops=dict(arrowstyle="->", color=PALETTE[5], alpha=arrow_alpha, lw=0.5),
            )
        ax.scatter(before[:, 0], before[:, 1], s=point_size, c=PALETTE[0], label="Before", zorder=5)
        ax.scatter(after[:, 0], after[:, 1], s=point_size, c=PALETTE[1], label="After", marker="x", zorder=5)
    else:
        unique_labels = np.unique(labels)
        for j, label in enumerate(unique_labels):
            mask = labels == label
            color = PALETTE[j % len(PALETTE)]
            name = label_names[j] if label_names else str(label)

            for i in np.where(mask)[0]:
                ax.annotate(
                    "",
                    xy=after[i],
                    xytext=before[i],
                    arrowprops=dict(arrowstyle="->", color=color, alpha=arrow_alpha, lw=0.5),
                )
            ax.scatter(before[mask, 0], before[mask, 1], s=point_size, c=color, label=f"{name} (before)", zorder=5)
            ax.scatter(after[mask, 0], after[mask, 1], s=point_size, c=color, marker="x", alpha=0.6, zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7)

    return fig, ax
