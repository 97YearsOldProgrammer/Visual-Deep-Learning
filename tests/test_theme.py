"""Tests for lib.theme."""

import matplotlib as mpl

from lib.theme import COLORS, PALETTE, apply_theme, get_color, get_fig


def test_apply_theme_sets_rcparams():
    apply_theme()
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
    assert mpl.rcParams["savefig.dpi"] == 300


def test_palette_length():
    assert len(PALETTE) == len(COLORS)


def test_get_color():
    assert get_color("blue") == "#2274A5"


def test_get_fig_returns_tuple():
    fig, ax = get_fig()
    assert fig is not None
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
