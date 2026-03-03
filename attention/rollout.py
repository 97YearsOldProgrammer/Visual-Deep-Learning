"""Attention rollout computation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def attention_rollout(
    attention_matrices: list[npt.NDArray[np.floating]],
    *,
    head_reduction: str = "mean",
    discard_ratio: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute attention rollout across transformer layers.

    Implements Abnar & Zuidema (2020) attention rollout:
    multiply attention matrices layer by layer, adding residual
    connections (identity matrix).

    Args:
        attention_matrices: List of (H, S, S) or (S, S) attention
            matrices, one per layer. H = heads, S = sequence length.
        head_reduction: How to reduce heads: "mean", "max", or "min".
        discard_ratio: Fraction of lowest attention weights to zero out
            per layer before rollout.

    Returns:
        (S, S) rollout attention matrix.
    """
    result = None

    for attn in attention_matrices:
        # Reduce heads if multi-head
        if attn.ndim == 3:
            if head_reduction == "mean":
                attn = attn.mean(axis=0)
            elif head_reduction == "max":
                attn = attn.max(axis=0)
            elif head_reduction == "min":
                attn = attn.min(axis=0)
            else:
                raise ValueError(f"Unknown head_reduction: {head_reduction}")

        # Discard low-attention entries
        if discard_ratio > 0:
            flat = attn.flatten()
            threshold = np.quantile(flat, discard_ratio)
            attn = np.where(attn < threshold, 0, attn)
            # Re-normalize rows
            row_sums = attn.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            attn = attn / row_sums

        # Add residual connection
        eye = np.eye(attn.shape[0])
        attn = 0.5 * attn + 0.5 * eye

        # Re-normalize
        attn = attn / attn.sum(axis=1, keepdims=True)

        if result is None:
            result = attn
        else:
            result = result @ attn

    return result
