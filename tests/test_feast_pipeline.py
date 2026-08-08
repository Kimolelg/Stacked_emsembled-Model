"""Unit tests for Feast table build (no feast materialize required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.feast_pipeline import (  # noqa: E402
    DOMAIN_PREFIXES,
    _domain_means,
    write_feast_artifacts,
)


def test_domain_means_from_engineered_columns():
    X = pd.DataFrame(
        {
            "Workload_A1": [1.0, 5.0],
            "Workload_A2": [3.0, 5.0],
            "WLB_D1": [2.0, 4.0],
            "Fin_A1": [3.0, 3.0],
            "Fin_D4": [1.0, 0.0],  # binary — excluded from fin_mean
            "Soc_A1": [4.0, 2.0],
        }
    )
    means = _domain_means(X)
    assert np.isclose(means.loc[0, "workload_mean"], 2.0)
    assert np.isclose(means.loc[1, "workload_mean"], 5.0)
    assert "fin_mean" in means.columns
    # Fin_D4 not averaged into fin_mean
    assert np.isclose(means.loc[0, "fin_mean"], 3.0)


def test_write_feast_artifacts(tmp_path, monkeypatch):
    import src.data.feast_pipeline as fp

    monkeypatch.setattr(fp, "FEAST_DATA", tmp_path)
    monkeypatch.setattr(fp, "PARQUET_PATH", tmp_path / "teacher_survey_features.parquet")
    monkeypatch.setattr(fp, "SCHEMA_PATH", tmp_path / "feature_schema.json")
    monkeypatch.setattr(fp, "LABELS_PATH", tmp_path / "teacher_labels.parquet")
    monkeypatch.setattr(fp, "ARTIFACTS", tmp_path / "artifacts")

    n = 5
    feature_order = ["Workload_A1", "WLB_D1", "Soc_A1"]
    feast_df = pd.DataFrame(
        {
            "teacher_id": np.arange(1, n + 1),
            "event_timestamp": pd.Timestamp.utcnow(),
            "created_timestamp": pd.Timestamp.utcnow(),
            "workload_mean": 3.0,
            "learners_mean": 3.0,
            "wlb_mean": 3.0,
            "perf_mean": 3.0,
            "time_mean": 3.0,
            "emot_mean": 3.0,
            "fin_mean": 3.0,
            "soc_mean": 3.0,
            "tech_awareness": 0.0,
            "fin_d4": 0.0,
            "phq_total": 10.0,
            "high_risk": 0,
            "Workload_A1": 4.0,
            "WLB_D1": 3.0,
            "Soc_A1": 2.0,
        }
    )
    schema = write_feast_artifacts(feast_df, feature_order)
    assert schema["n_features"] == 3
    assert (tmp_path / "teacher_survey_features.parquet").exists()
    assert (tmp_path / "feature_schema.json").exists()
    assert (tmp_path / "artifacts" / "feature_order.joblib").exists()
