"""Publication-quality matplotlib theme for deep learning figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Color palette: accessible, print-safe, distinct
COLORS = {
    "blue": "#2274A5",
    "orange": "#E36414",
    "green": "#32936F",
    "red": "#C1292E",
    "purple": "#6B4C9A",
    "gray": "#6C757D",
}

PALETTE = list(COLORS.values())


def apply_theme(
    *,
    font_size: int = 10,
    fig_width: float = 3.5,
    fig_height: float = 2.6,
    dpi: int = 300,
    font_family: str = "serif",
    usetex: bool = False,
) -> None:
    """Apply publication-quality rcParams globally.

    Default figure size targets single-column width (~3.5 in).
    """
    mpl.rcParams.update(
        {
            # Font
            "font.family": font_family,
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            # Figure
            "figure.figsize": (fig_width, fig_height),
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Axes
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "axes.grid": False,
            # Ticks
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # Lines
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            # Legend
            "legend.frameon": False,
            "legend.borderaxespad": 0.3,
            # LaTeX
            "text.usetex": usetex,
        }
    )


def get_color(name: str) -> str:
    """Get a named color from the palette."""
    return COLORS[name]


def get_fig(
    nrows: int = 1,
    ncols: int = 1,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure with the publication theme applied."""
    fig, ax = plt.subplots(nrows, ncols, **kwargs)
    return fig, ax
