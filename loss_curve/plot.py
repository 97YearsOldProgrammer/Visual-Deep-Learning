"""Training/loss curve plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from lib.smooth import ema
from lib.theme import PALETTE


def plot_loss(
    train: npt.NDArray[np.floating],
    val: npt.NDArray[np.floating] | None = None,
    *,
    smooth: bool = True,
    alpha: float = 0.1,
    xlabel: str = "Step",
    ylabel: str = "Loss",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot training (and optionally validation) loss curves.

    Args:
        train: 1-D array of training loss values.
        val: Optional 1-D array of validation loss values.
        smooth: Whether to apply EMA smoothing.
        alpha: EMA smoothing factor (smaller = smoother).
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Optional figure title.
        ax: Optional existing Axes to plot on.

    Returns:
        (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    steps = np.arange(len(train))

    # Raw data as faint background
    if smooth:
        ax.plot(steps, train, color=PALETTE[0], alpha=0.2, linewidth=0.5)
        ax.plot(steps, ema(train, alpha), color=PALETTE[0], label="Train")
    else:
        ax.plot(steps, train, color=PALETTE[0], label="Train")

    if val is not None:
        val_steps = np.linspace(0, len(train) - 1, len(val))
        if smooth:
            ax.plot(val_steps, val, color=PALETTE[1], alpha=0.2, linewidth=0.5)
            ax.plot(val_steps, ema(val, alpha), color=PALETTE[1], label="Val")
        else:
            ax.plot(val_steps, val, color=PALETTE[1], label="Val")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()

    return fig, ax


def plot_metric_comparison(
    metrics: dict[str, npt.NDArray[np.floating]],
    *,
    smooth: bool = True,
    alpha: float = 0.1,
    xlabel: str = "Step",
    ylabel: str = "Value",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple metrics on the same axes.

    Args:
        metrics: Dict mapping metric name to 1-D array.
        smooth: Whether to apply EMA smoothing.
        alpha: EMA smoothing factor.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Optional figure title.
        ax: Optional existing Axes.

    Returns:
        (fig, ax) tuple.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for i, (name, values) in enumerate(metrics.items()):
        color = PALETTE[i % len(PALETTE)]
        steps = np.arange(len(values))
        if smooth:
            ax.plot(steps, values, color=color, alpha=0.2, linewidth=0.5)
            ax.plot(steps, ema(values, alpha), color=color, label=name)
        else:
            ax.plot(steps, values, color=color, label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()

    return fig, ax
