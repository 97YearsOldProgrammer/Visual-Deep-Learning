"""Tests for lib.io."""

import json
import tempfile
from pathlib import Path

import numpy as np

from lib.io import load_csv, load_json, load_jsonl


def test_load_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("step,loss,acc\n")
        f.write("1,0.5,0.8\n")
        f.write("2,0.3,0.9\n")
        path = f.name

    data = load_csv(path)
    assert "loss" in data
    np.testing.assert_array_almost_equal(data["loss"], [0.5, 0.3])
    Path(path).unlink()


def test_load_csv_with_columns():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("step,loss,acc\n")
        f.write("1,0.5,0.8\n")
        path = f.name

    data = load_csv(path, columns=["loss"])
    assert "loss" in data
    assert "acc" not in data
    Path(path).unlink()


def test_load_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"metrics": [1, 2, 3]}, f)
        path = f.name

    data = load_json(path, key="metrics")
    assert data == [1, 2, 3]
    Path(path).unlink()


def test_load_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")
        f.write(json.dumps({"step": 2, "loss": 0.3}) + "\n")
        path = f.name

    data = load_jsonl(path)
    assert "loss" in data
    np.testing.assert_array_almost_equal(data["loss"], [0.5, 0.3])
    Path(path).unlink()
