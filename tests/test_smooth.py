"""Tests for lib.smooth."""

import numpy as np
import pytest

from lib.smooth import ema, rolling_mean, savgol


@pytest.fixture
def noisy_signal():
    rng = np.random.default_rng(0)
    return np.sin(np.linspace(0, 4 * np.pi, 100)) + rng.normal(0, 0.2, 100)


def test_ema_shape(noisy_signal):
    result = ema(noisy_signal, alpha=0.1)
    assert result.shape == noisy_signal.shape


def test_ema_first_value(noisy_signal):
    result = ema(noisy_signal, alpha=0.1)
    assert result[0] == noisy_signal[0]


def test_ema_smoother_with_small_alpha(noisy_signal):
    smooth_01 = ema(noisy_signal, alpha=0.1)
    smooth_05 = ema(noisy_signal, alpha=0.5)
    # Smaller alpha → smoother → lower variance
    assert np.std(np.diff(smooth_01)) < np.std(np.diff(smooth_05))


def test_rolling_mean_shape(noisy_signal):
    result = rolling_mean(noisy_signal, window=10)
    assert result.shape == noisy_signal.shape


def test_savgol_shape(noisy_signal):
    result = savgol(noisy_signal, window=11, polyorder=3)
    assert result.shape == noisy_signal.shape


def test_savgol_short_input():
    short = np.array([1.0, 2.0, 3.0])
    result = savgol(short, window=11, polyorder=3)
    assert result.shape == short.shape
