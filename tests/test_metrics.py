"""Pruebas del evaluador de AP (python -m pytest tests -q)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import evaluate  # noqa: E402


def _gt(rows):
    return pd.DataFrame(rows, columns=["image_id", "category", "x1", "y1", "x2", "y2", "ignore"])


def _det(rows):
    return pd.DataFrame(rows, columns=["image_id", "category", "x1", "y1", "x2", "y2", "score"])


def test_perfect():
    gt = _gt([("a", "c", 0, 0, 10, 10, False), ("b", "c", 5, 5, 20, 20, False)])
    det = _det([("a", "c", 0, 0, 10, 10, .9), ("b", "c", 5, 5, 20, 20, .8)])
    r = evaluate(gt, det, ["a", "b"], ["c"]).iloc[0]
    assert np.isclose(r.AP50, 1) and np.isclose(r.AP50_95, 1)


def test_fp_ranked_first_halves_precision():
    gt = _gt([("a", "c", 0, 0, 10, 10, False)])
    det = _det([("a", "c", 50, 50, 60, 60, .9), ("a", "c", 0, 0, 10, 10, .8)])
    r = evaluate(gt, det, ["a"], ["c"]).iloc[0]
    assert np.isclose(r.AP50, 0.5)


def test_ignored_gt_does_not_create_fp():
    gt = _gt([("a", "c", 0, 0, 10, 10, False), ("a", "c", 50, 50, 60, 60, True)])
    det = _det([("a", "c", 50, 50, 60, 60, .95), ("a", "c", 0, 0, 10, 10, .8)])
    r = evaluate(gt, det, ["a"], ["c"]).iloc[0]
    assert np.isclose(r.AP50, 1) and r.n_gt == 1


def test_missing_class_is_zero():
    gt = _gt([("a", "c", 0, 0, 10, 10, False)])
    r = evaluate(gt, _det([]), ["a"], ["c"]).iloc[0]
    assert r.AP50 == 0
