"""Smoke tests for Meta-Model 4 (in meta_models.py) on synthetic data."""

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
pytest.importorskip("imblearn")

from src.models.meta_models import (  # noqa: E402
    MetaModel4,
    MetaModel4Config,
    evaluate_binary_model,
    metrics_for_mlflow,
    train_meta_model4,
)


def _synthetic_xy(n: int = 120, n_features: int = 12, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    logits = X["f0"] + 0.5 * X["f1"] - 0.3 * X["f2"]
    y = (logits > logits.median()).astype(int)
    phq = pd.Series(
        np.clip(
            np.round(
                10
                + 8
                * (logits - logits.min())
                / (logits.max() - logits.min() + 1e-9)
            ),
            0,
            27,
        ),
        name="PHQ_Total",
    )
    phq.iloc[:10] = 15.0
    return X, y, phq


@pytest.mark.slow
def test_train_meta_model4_smoke():
    X, y, phq = _synthetic_xy()
    feature_order = list(X.columns)
    cfg = MetaModel4Config(
        random_state=42,
        rf_n_estimators=20,
        rf_max_depth=4,
        xgb_n_estimators=20,
        xgb_max_depth=2,
        cat_iterations=30,
        cat_depth=3,
        cat_loss_function="Logloss",
        cal_cv=2,
        meta_n_estimators=20,
        meta_max_depth=2,
    )
    model, info = train_meta_model4(
        X_train=X,
        y_train=y,
        phq_train=phq,
        feature_order=feature_order,
        cfg=cfg,
    )
    assert isinstance(model, MetaModel4)
    assert info["resampled_n"] > 0
    proba = model.predict_proba(X)
    pred = model.predict(X)
    assert proba.shape == (len(X), 2)
    assert pred.shape == (len(X),)

    metrics = evaluate_binary_model(model, X, y)
    for key in (
        "accuracy",
        "precision",
        "recall",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "roc_auc",
    ):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    flat = metrics_for_mlflow("test", metrics)
    assert "test_accuracy" in flat
    assert "test_precision" in flat
    assert "test_recall" in flat


def test_registered_names_are_equal():
    from src.models.meta_models import REGISTERED_MODEL_NAMES

    assert REGISTERED_MODEL_NAMES["meta4"] == "teacher-mental-health-risk-meta4"
    assert "production" not in REGISTERED_MODEL_NAMES["meta4"].lower()
    for k in ("meta1", "meta2", "meta3", "meta4"):
        assert k in REGISTERED_MODEL_NAMES[k]
