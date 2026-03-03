"""Smoothing utilities for noisy training curves."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter


def ema(
    values: npt.NDArray[np.floating],
    alpha: float = 0.1,
) -> npt.NDArray[np.floating]:
    """Exponential moving average.

    Args:
        values: 1-D array of values.
        alpha: Smoothing factor in (0, 1]. Smaller = smoother.

    Returns:
        Smoothed array of the same shape.
    """
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def rolling_mean(
    values: npt.NDArray[np.floating],
    window: int = 10,
) -> npt.NDArray[np.floating]:
    """Rolling average with edge-aware padding.

    Uses 'reflect' mode so output length matches input.
    """
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def savgol(
    values: npt.NDArray[np.floating],
    window: int = 11,
    polyorder: int = 3,
) -> npt.NDArray[np.floating]:
    """Savitzky-Golay filter.

    Good for preserving peaks while removing noise.
    """
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window < polyorder + 2:
        return values.copy()
    return savgol_filter(values, window, polyorder)
