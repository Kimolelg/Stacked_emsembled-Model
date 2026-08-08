"""Tests for runtime + batch data validation (mental health)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_validation import (  # noqa: E402
    validate_batch_features,
    validate_feature_vector,
    validate_questionnaire,
)


def test_questionnaire_rejects_bad_section_score():
    r = validate_questionnaire({"workload": 9.0, "gender": "Female"})
    assert r["valid"] is False
    assert any("workload" in e for e in r["errors"])


def test_questionnaire_rejects_negative_teacher_id():
    r = validate_questionnaire({"teacher_id": -1})
    assert r["valid"] is False


def test_questionnaire_accepts_valid_live_form():
    r = validate_questionnaire(
        {
            "gender": "Female",
            "age_category": "35-44",
            "workload": 4.0,
            "emotional_impact": 4.5,
        }
    )
    assert r["valid"] is True


def test_questionnaire_empty_warns_but_valid():
    r = validate_questionnaire({})
    assert r["valid"] is True
    assert r["warnings"]


def test_feature_vector_likert_range():
    fo = ["Workload_A1", "WLB_D1", "Tech_Awareness"]
    bad = {"Workload_A1": 9.0, "WLB_D1": 3.0, "Tech_Awareness": 0.0}
    r = validate_feature_vector(bad, fo)
    assert r["valid"] is False

    good = {"Workload_A1": 4.0, "WLB_D1": 3.0, "Tech_Awareness": 1.0}
    r2 = validate_feature_vector(good, fo)
    assert r2["valid"] is True


def test_batch_validation_synthetic():
    n = 50
    df = pd.DataFrame(
        {
            "teacher_id": np.arange(1, n + 1),
            "Workload_A1": np.full(n, 3.0),
            "WLB_D1": np.full(n, 4.0),
            "Tech_Awareness": np.zeros(n),
            "high_risk": np.random.default_rng(0).integers(0, 2, n),
            "phq_total": np.full(n, 12.0),
        }
    )
    r = validate_batch_features(
        df, feature_order=["Workload_A1", "WLB_D1", "Tech_Awareness"]
    )
    assert r["total"] > 0
    assert "engine" in r
    # Should largely pass
    assert r["pass_rate"] >= 0.5
