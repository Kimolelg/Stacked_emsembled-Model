"""Tests for simplified questionnaire → full feature expansion."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.feature_builder import (  # noqa: E402
    build_default_features,
    expand_questionnaire_to_features,
    features_dict_with_defaults,
)


@pytest.fixture
def feature_order():
    return [
        "Workload_A1",
        "Workload_A2",
        "Learners_B1",
        "WLB_D1",
        "Fin_A1",
        "Fin_D4",
        "Soc_A1",
        "Tech_Awareness",
        "Tech_Platforms_Not_applicable",
        "Age_Category_25_34_",
        "Age_Category_35_44",
        "Gender_Female",
        "Gender_Male_",
        "School_Type_Primary_School",
        "School_Type_Secondary_School",
    ]


def test_defaults_neutral_likert(feature_order):
    d = build_default_features(feature_order)
    assert d["Workload_A1"] == 3.0
    assert d["Fin_D4"] == 0.0
    assert d["Tech_Awareness"] == 0.0
    assert d["Tech_Platforms_Not_applicable"] == 1.0


def test_section_broadcast(feature_order):
    feats, provided = expand_questionnaire_to_features(
        feature_order,
        workload=4.0,
        gender="Female",
        age_category="25-34",
    )
    assert feats["Workload_A1"] == 4.0
    assert feats["Workload_A2"] == 4.0
    assert feats["Learners_B1"] == 3.0  # default
    assert feats["Gender_Female"] == 1.0
    assert feats["Gender_Male_"] == 0.0
    assert "workload" in provided
    assert "gender" in provided


def test_partial_features_no_error(feature_order):
    full = features_dict_with_defaults(feature_order, {"Workload_A1": 5.0})
    assert full["Workload_A1"] == 5.0
    assert full["Workload_A2"] == 3.0
    assert len(full) == len(feature_order)
