"""Tests for prediction logging and drift monitor (KS path; Evidently optional)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.monitoring.drift_monitor import DriftMonitor, _ks_drift  # noqa: E402
from src.monitoring.prediction_logger import PredictionLogger  # noqa: E402


def test_prediction_logger_roundtrip(tmp_path):
    log_path = tmp_path / "prod.parquet"
    plog = PredictionLogger(log_path=log_path)
    features = {
        "Workload_A1": 4.0,
        "Workload_A2": 5.0,
        "WLB_D1": 3.0,
        "Emot_C3": 4.0,
        "Tech_Awareness": 0.0,
    }
    plog.log_prediction(
        features=features,
        prediction=1,
        prediction_proba=0.82,
        model_name="test",
        teacher_id=7,
    )
    assert plog.count() == 1
    df = plog.load_production()
    assert "workload_mean" in df.columns
    assert df.iloc[0]["prediction"] == 1
    assert abs(df.iloc[0]["prediction_proba"] - 0.82) < 1e-6


def test_ks_drift_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(
        {
            "workload_mean": rng.normal(3.0, 0.3, 200),
            "wlb_mean": rng.normal(3.0, 0.3, 200),
            "emot_mean": rng.normal(3.0, 0.3, 200),
        }
    )
    cur = pd.DataFrame(
        {
            "workload_mean": rng.normal(4.5, 0.3, 200),  # shifted
            "wlb_mean": rng.normal(3.0, 0.3, 200),
            "emot_mean": rng.normal(4.2, 0.3, 200),  # shifted
        }
    )
    res = _ks_drift(ref, cur, ["workload_mean", "wlb_mean", "emot_mean"])
    assert res["n_checked"] >= 2
    assert res["columns"]["workload_mean"]["drift_detected"] is True


def test_drift_monitor_insufficient_current(tmp_path, monkeypatch):
    mon = DriftMonitor(min_current_rows=50)
    # tiny reference
    ref = pd.DataFrame(
        {
            "workload_mean": [3.0] * 100,
            "wlb_mean": [3.0] * 100,
            "emot_mean": [3.0] * 100,
        }
    )
    cur = pd.DataFrame(
        {
            "workload_mean": [4.0] * 10,
            "wlb_mean": [4.0] * 10,
            "emot_mean": [4.0] * 10,
        }
    )
    out = mon.check_drift(current=cur, reference=ref)
    assert out["success"] is False
    assert "Need at least" in out["reason"]
