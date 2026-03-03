"""Tests for attention.rollout."""

import numpy as np
from scipy.special import softmax

from attention.rollout import attention_rollout


def _make_attention(n_heads: int, seq_len: int, n_layers: int):
    rng = np.random.default_rng(42)
    layers = []
    for _ in range(n_layers):
        raw = rng.normal(0, 1, (n_heads, seq_len, seq_len))
        layers.append(softmax(raw, axis=-1))
    return layers


def test_rollout_shape():
    layers = _make_attention(4, 8, 3)
    result = attention_rollout(layers)
    assert result.shape == (8, 8)


def test_rollout_rows_sum_to_one():
    layers = _make_attention(4, 8, 3)
    result = attention_rollout(layers)
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)


def test_rollout_single_layer():
    layers = _make_attention(1, 4, 1)
    result = attention_rollout(layers)
    assert result.shape == (4, 4)
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)


def test_rollout_2d_input():
    """Test with pre-reduced (S, S) attention matrices."""
    rng = np.random.default_rng(42)
    layers = [softmax(rng.normal(0, 1, (6, 6)), axis=-1) for _ in range(3)]
    result = attention_rollout(layers)
    assert result.shape == (6, 6)


def test_rollout_discard_ratio():
    layers = _make_attention(4, 8, 3)
    result = attention_rollout(layers, discard_ratio=0.5)
    assert result.shape == (8, 8)
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)
