"""Smoke tests for Meta-Models 1–3 (tiny synthetic data)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("sklearn")
pytest.importorskip("xgboost")
pytest.importorskip("catboost")
pytest.importorskip("lightgbm")
pytest.importorskip("imblearn")

from src.models.meta_models import (  # noqa: E402
    train_meta_model1,
    train_meta_model2,
    train_meta_model3,
)


def _xy(n=100, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series((X["f0"] + 0.5 * X["f1"] > 0).astype(int))
    return X, y


@pytest.mark.slow
@pytest.mark.parametrize(
    "trainer",
    [train_meta_model1, train_meta_model2],
    ids=["meta1", "meta2"],
)
def test_meta12_smoke(trainer):
    X, y = _xy()
    # Use small subsample of fit by monkeypatching would be hard;
    # Stacking with cv=5 needs enough rows — n=100 is OK but slow.
    # Speed: we rely on default notebook configs; allow ~1–2 min total.
    result = trainer(X, y, random_state=0)
    assert result.model is not None
    proba = result.model.predict_proba(X.head(5))
    pred = result.model.predict(X.head(5))
    assert proba.shape == (5, 2) or proba.shape[0] == 5
    assert pred.shape[0] == 5


@pytest.mark.slow
def test_meta3_threshold():
    X, y = _xy(n=120)
    X_tr, X_te = X.iloc[:90], X.iloc[90:]
    y_tr, y_te = y.iloc[:90], y.iloc[90:]
    result = train_meta_model3(
        X_tr, y_tr, X_test=X_te, y_test=y_te, random_state=0, optimize_threshold=True
    )
    assert 0.4 <= result.threshold <= 0.6
    pred = result.model.predict(X_te)
    assert pred.shape[0] == len(X_te)
