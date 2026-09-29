"""MLflow tracking + registry smoke tests (file store — no server required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("mlflow")
pytest.importorskip("sklearn")

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from src.utils.mlflow_tracking import (  # noqa: E402
    get_model_info,
    load_model_from_registry,
    log_training_run,
    resolve_registered_model_name,
)


@pytest.fixture
def file_tracking_uri(tmp_path):
    return (tmp_path / "mlruns").as_uri()


def test_resolve_base_plus_variant():
    assert (
        resolve_registered_model_name("teacher-mental-health-risk", "meta4")
        == "teacher-mental-health-risk-meta4"
    )
    assert (
        resolve_registered_model_name("teacher-mental-health-risk", "meta1")
        == "teacher-mental-health-risk-meta1"
    )


def test_resolve_already_full_name():
    assert (
        resolve_registered_model_name("teacher-mental-health-risk-meta3", "meta4")
        == "teacher-mental-health-risk-meta3"
    )


def test_resolve_from_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_REGISTERED_MODEL_NAME", "teacher-mental-health-risk")
    monkeypatch.setenv("MLFLOW_MODEL_VARIANT", "rf")
    assert resolve_registered_model_name() == "teacher-mental-health-risk-rf"


def test_resolve_base_without_variant(monkeypatch):
    monkeypatch.setenv("MLFLOW_REGISTERED_MODEL_NAME", "teacher-mental-health-risk")
    monkeypatch.setenv("MLFLOW_MODEL_VARIANT", "")
    assert resolve_registered_model_name() == "teacher-mental-health-risk"


@pytest.mark.slow
def test_log_and_load_without_auto_champion(file_tracking_uri, tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"] + X["f1"] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
    model.fit(X, y)

    import joblib

    art_model = tmp_path / "model.joblib"
    art_fo = tmp_path / "feature_order.joblib"
    joblib.dump(model, art_model)
    joblib.dump(list(X.columns), art_fo)

    reg_name = "test-teacher-mh-model"
    info = log_training_run(
        model=model,
        feature_order=list(X.columns),
        params={"model_type": "RF", "n_estimators": 10},
        metrics={
            "test_accuracy": 0.9,
            "test_precision": 0.85,
            "test_recall": 0.8,
            "test_macro_f1": 0.88,
            "test_roc_auc": 0.95,
        },
        tags={"unit_test": "true"},
        artifact_paths={"model": art_model, "feature_order": art_fo},
        registered_model_name=reg_name,
        run_name="unit_test_run",
        input_example=X.head(2),
        set_champion=False,
        set_challenger=False,
        tracking_uri=file_tracking_uri,
        experiment_name="unit-test-experiment",
    )

    assert info["run_id"]
    assert info["version"] is not None
    # No production-style aliases auto-assigned in code
    assert "champion" not in info["aliases"]
    assert "challenger" not in info["aliases"]

    # Resolve highest version explicitly (not used by serving API)
    loaded = load_model_from_registry(
        model_name=reg_name,
        alias="latest",
        tracking_uri=file_tracking_uri,
    )
    assert loaded.predict_proba(X.head(3)).shape[0] == 3
    assert loaded.predict(X.head(3)).shape[0] == 3
