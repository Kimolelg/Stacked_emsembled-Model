"""
Unit tests for Week 1-2 data processing pipeline.

Validates alignment with mental_health_final_model.py critical behaviors:
- Fin_D4 is Yes/No binary (not Likert 1-5)
- Label-leakage depression-type dummies removed after multi-select encoding
- Multicollinear columns dropped
- Sub_County one-hot encoded
- High_Risk = PHQ_Total >= 15
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.processing import (  # noqa: E402
    MULTICOLLINEAR_DROP,
    engineer_features_and_target,
    get_feature_order,
    prepare_for_modeling,
)


def _synthetic_frame(n: int = 24) -> pd.DataFrame:
    likert = [
        "Strongly Disagree",
        "Disagree",
        "Neutral",
        "Agree",
        "Strongly Agree",
    ]
    phq = [
        "Not at all",
        "Several days",
        "More than half the day",
        "Nearly every day",
    ]
    rows = []
    for i in range(n):
        rows.append(
            {
                "Age_Category": "25-34",
                "Gender": "Female" if i % 2 == 0 else "Male",
                "Years_in_Service": "6-10",
                "School_Type": "Secondary",
                "Sub_County": "A" if i < n // 2 else "B",
                "County": "Nairobi",
                "Education_Qualification": "Degree",
                "ICT_Skills": "Intermediate",
                "Depression_Experience": "Yes" if i % 3 == 0 else "No",
                "Depression_Types": (
                    "None of the above;Burnout Related Depression "
                    "(Due to prolonged work-related stress or exhaustion)"
                    if i % 3 == 0
                    else "None of the above"
                ),
                "Tech_Awareness": "Yes" if i % 2 else "No",
                "Tech_Platforms": "Not applicable",
                "Workload_A1": likert[i % 5],
                "Learners_B1": likert[(i + 1) % 5],
                "Learners_B4": likert[(i + 2) % 5],
                "WLB_D1": likert[(i + 3) % 5],
                "Perf_A1": likert[i % 5],
                "Perf_A2": likert[(i + 1) % 5],
                "Perf_A5": likert[(i + 2) % 5],
                "Time_B2": likert[i % 5],
                "Time_B5": likert[(i + 1) % 5],
                "Emot_C1": likert[i % 5],
                "Emot_C2": likert[(i + 2) % 5],
                "Emot_C5": likert[(i + 3) % 5],
                "Fin_A1": likert[i % 5],
                "Fin_D4": "Yes" if i % 4 == 0 else "No",
                "Soc_A1": likert[i % 5],
                "PHQ1": phq[i % 4],
                "PHQ2": phq[(i + 1) % 4],
                "PHQ3": phq[(i + 2) % 4],
                "PHQ4": phq[i % 4],
                "PHQ5": phq[(i + 1) % 4],
                "PHQ6": phq[(i + 2) % 4],
                "PHQ7": phq[i % 4],
                "PHQ8": phq[(i + 1) % 4],
                "PHQ9": phq[(i + 2) % 4],
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def engineered():
    return engineer_features_and_target(_synthetic_frame())


def test_high_risk_definition(engineered):
    assert (
        (engineered["PHQ_Total"] >= 15).astype(int) == engineered["High_Risk"]
    ).all()


def test_fin_d4_is_binary_not_likert(engineered):
    vals = set(engineered["Fin_D4"].unique().tolist())
    assert vals.issubset({0.0, 1.0})


def test_multicollinear_dropped(engineered):
    for col in MULTICOLLINEAR_DROP:
        assert col not in engineered.columns


def test_leakage_removed(engineered):
    features = get_feature_order(engineered)
    assert "Depression_Experience" not in features
    assert not any(c.startswith("Depression_Types_") for c in features)


def test_sub_county_ohe(engineered):
    features = get_feature_order(engineered)
    assert any(c.startswith("Sub_County_") for c in features)


def test_phq_items_not_features(engineered):
    features = get_feature_order(engineered)
    assert not any(c.startswith("PHQ") and c != "PHQ_Total" for c in features)
    assert "PHQ_Total" not in features  # excluded from model features


def test_prepare_for_modeling_numeric(engineered):
    X, y, feature_order = prepare_for_modeling(engineered)
    assert y is not None
    assert len(feature_order) == X.shape[1]
    assert X.dtypes.apply(lambda t: np.issubdtype(t, np.number)).all()
    assert "High_Risk" not in feature_order
    assert "Depression_Severity" not in feature_order
