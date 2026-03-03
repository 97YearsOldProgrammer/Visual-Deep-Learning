"""IO utilities for loading training logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def load_csv(
    path: str | Path,
    columns: list[str] | None = None,
) -> dict[str, npt.NDArray[np.floating]]:
    """Load numeric columns from a CSV file.

    Args:
        path: Path to CSV file.
        columns: Column names to load. If None, loads all numeric columns.

    Returns:
        Dict mapping column name to numpy array.
    """
    path = Path(path)
    data: dict[str, list[float]] = {}

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if columns is not None and key not in columns:
                    continue
                try:
                    data.setdefault(key, []).append(float(val))
                except (ValueError, TypeError):
                    continue

    return {k: np.array(v) for k, v in data.items()}


def load_json(
    path: str | Path,
    key: str | None = None,
) -> dict[str, Any] | list[Any]:
    """Load a JSON file.

    Args:
        path: Path to JSON file.
        key: Optional top-level key to extract.

    Returns:
        Parsed JSON content.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def load_jsonl(
    path: str | Path,
    fields: list[str] | None = None,
) -> dict[str, npt.NDArray[np.floating]]:
    """Load numeric fields from a JSON Lines file.

    Each line is a JSON object (e.g., one per training step).

    Args:
        path: Path to JSONL file.
        fields: Field names to extract. If None, extracts all numeric fields.

    Returns:
        Dict mapping field name to numpy array.
    """
    path = Path(path)
    data: dict[str, list[float]] = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for key, val in obj.items():
                if fields is not None and key not in fields:
                    continue
                if isinstance(val, (int, float)):
                    data.setdefault(key, []).append(float(val))

    return {k: np.array(v) for k, v in data.items()}
