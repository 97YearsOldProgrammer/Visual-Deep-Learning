"""Procrustes alignment for comparing embedding spaces."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial import procrustes


def procrustes_align(
    before: npt.NDArray[np.floating],
    after: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float]:
    """Align two embedding matrices using Procrustes analysis.

    Finds the optimal rotation/reflection/scaling to minimize
    the sum of squared differences between the two point sets.

    Both inputs should already be 2-D (e.g., after t-SNE or UMAP).

    Args:
        before: (N, 2) array of "before" coordinates.
        after: (N, 2) array of "after" coordinates.

    Returns:
        (aligned_before, aligned_after, disparity) tuple.
        disparity is the sum of squared errors after alignment.
    """
    aligned_before, aligned_after, disparity = procrustes(before, after)
    return aligned_before, aligned_after, disparity
