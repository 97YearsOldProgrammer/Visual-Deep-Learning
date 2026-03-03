"""t-SNE embedding map plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.manifold import TSNE

from lib.theme import PALETTE


def compute_tsne(
    embeddings: npt.NDArray[np.floating],
    *,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> npt.NDArray[np.floating]:
    """Run t-SNE on high-dimensional embeddings.

    Args:
        embeddings: (N, D) array of embeddings.
        perplexity: t-SNE perplexity parameter.
        n_iter: Number of optimization iterations.
        random_state: Random seed for reproducibility.

    Returns:
        (N, 2) array of 2-D coordinates.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
    )
    return tsne.fit_transform(embeddings)


def plot_tsne(
    embeddings: npt.NDArray[np.floating],
    labels: npt.NDArray[np.integer] | None = None,
    *,
    label_names: list[str] | None = None,
    perplexity: float = 30.0,
    point_size: float = 8,
    point_alpha: float = 0.6,
    title: str | None = None,
    ax: plt.Axes | None = None,
    precomputed_2d: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a t-SNE embedding map.

    Args:
        embeddings: (N, D) array. If precomputed_2d=True, must be (N, 2).
        labels: Optional (N,) integer class labels for coloring.
        label_names: Optional names for each class.
        perplexity: t-SNE perplexity (ignored if precomputed_2d).
        point_size: Scatter point size.
        point_alpha: Scatter point alpha.
        title: Optional figure title.
        ax: Optional existing Axes.
        precomputed_2d: If True, skip t-SNE and use embeddings directly.

    Returns:
        (fig, ax) tuple.
    """
    if precomputed_2d:
        coords = embeddings
    else:
        coords = compute_tsne(embeddings, perplexity=perplexity)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=point_size, alpha=point_alpha, c=PALETTE[0])
    else:
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            name = label_names[i] if label_names else str(label)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size,
                alpha=point_alpha,
                c=PALETTE[i % len(PALETTE)],
                label=name,
            )
        ax.legend(markerscale=2)

    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    return fig, ax
