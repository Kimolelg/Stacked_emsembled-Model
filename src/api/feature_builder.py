"""
Map simplified questionnaire inputs → full engineered feature vectors.

Choice rationale
----------------
Models are trained on ~200 item-level features. Requiring every column at
inference is impractical. We keep the trained models unchanged and:

1. Accept a **compact screening questionnaire** (demographics + section scores).
2. Expand section scores onto all Likert items in that domain (broadcast).
3. One-hot encode demographics to match columns present in feature_order.
4. Fill anything still missing with **safe defaults**:
   - Likert items → 3.0 (Neutral)
   - Binary / one-hot / other → 0.0

Users may still pass partial `features` overrides. Missing keys never 422.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Survey domains → column name prefixes in engineered data
SECTION_PREFIXES: Dict[str, str] = {
    "workload": "Workload_",
    "learners": "Learners_",
    "work_life_balance": "WLB_",
    "performance": "Perf_",
    "time_pressure": "Time_",
    "emotional_impact": "Emot_",
    "financial": "Fin_",
    "social_support": "Soc_",
}

# Binary columns that should not get Likert default 3.0
BINARY_COLUMNS = {"Tech_Awareness", "Fin_D4"}

# Demographic field → OHE column prefix in feature_order
DEMOGRAPHIC_PREFIXES: Dict[str, str] = {
    "age_category": "Age_Category_",
    "gender": "Gender_",
    "years_in_service": "Years_in_Service_",
    "school_type": "School_Type_",
    "education_qualification": "Education_Qualification_",
    "ict_skills": "ICT_Skills_",
}


def _norm(s: str) -> str:
    """Normalize labels for fuzzy OHE matching."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def is_likert_column(name: str) -> bool:
    if name in BINARY_COLUMNS:
        return False
    return any(name.startswith(p) for p in SECTION_PREFIXES.values())


def build_default_features(feature_order: Sequence[str]) -> Dict[str, float]:
    """Neutral / zero defaults for every model feature."""
    out: Dict[str, float] = {}
    for col in feature_order:
        if is_likert_column(col):
            out[col] = 3.0
        else:
            out[col] = 0.0
    # Prefer "Not applicable" tech platforms when nothing specified
    for col in feature_order:
        if "Tech_Platforms_Not_applicable" in col or col.endswith(
            "Tech_Platforms_Not_applicable"
        ):
            out[col] = 1.0
    return out


def _apply_section_score(
    features: Dict[str, float],
    feature_order: Sequence[str],
    prefix: str,
    score: float,
) -> None:
    score = float(np.clip(score, 1.0, 5.0))
    for col in feature_order:
        if not col.startswith(prefix):
            continue
        # Fin_D4 is Yes/No, not Likert
        if col == "Fin_D4":
            continue
        features[col] = score


def _apply_demographic_ohe(
    features: Dict[str, float],
    feature_order: Sequence[str],
    prefix: str,
    value: Optional[str],
) -> None:
    """Set matching one-hot column to 1; leave others 0 for that prefix."""
    if not value:
        return
    # Zero this demographic family first
    family = [c for c in feature_order if c.startswith(prefix)]
    for c in family:
        features[c] = 0.0

    target = _norm(value)
    best = None
    for c in family:
        suffix = c[len(prefix) :]
        if target and (target in _norm(suffix) or _norm(suffix) in target):
            best = c
            break
    if best is None:
        # fallback: exact-ish start match after stripping underscores
        for c in family:
            if target[:6] and target[:6] in _norm(c):
                best = c
                break
    if best is not None:
        features[best] = 1.0


def expand_questionnaire_to_features(
    feature_order: Sequence[str],
    *,
    age_category: Optional[str] = None,
    gender: Optional[str] = None,
    years_in_service: Optional[str] = None,
    school_type: Optional[str] = None,
    education_qualification: Optional[str] = None,
    ict_skills: Optional[str] = None,
    workload: Optional[float] = None,
    learners: Optional[float] = None,
    work_life_balance: Optional[float] = None,
    performance: Optional[float] = None,
    time_pressure: Optional[float] = None,
    emotional_impact: Optional[float] = None,
    financial: Optional[float] = None,
    social_support: Optional[float] = None,
    tech_awareness: Optional[bool] = None,
    financial_literacy_training: Optional[bool] = None,
    feature_overrides: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Build a full feature dict + list of columns that were user-specified
    (section broadcast counts as specified for reporting).
    """
    features = build_default_features(feature_order)
    specified: List[str] = []

    demos = {
        "age_category": age_category,
        "gender": gender,
        "years_in_service": years_in_service,
        "school_type": school_type,
        "education_qualification": education_qualification,
        "ict_skills": ict_skills,
    }
    for field, val in demos.items():
        if val is not None:
            _apply_demographic_ohe(
                features, feature_order, DEMOGRAPHIC_PREFIXES[field], val
            )
            specified.append(field)

    sections = {
        "workload": workload,
        "learners": learners,
        "work_life_balance": work_life_balance,
        "performance": performance,
        "time_pressure": time_pressure,
        "emotional_impact": emotional_impact,
        "financial": financial,
        "social_support": social_support,
    }
    for name, score in sections.items():
        if score is not None:
            _apply_section_score(
                features, feature_order, SECTION_PREFIXES[name], float(score)
            )
            specified.append(name)

    if tech_awareness is not None and "Tech_Awareness" in features:
        features["Tech_Awareness"] = 1.0 if tech_awareness else 0.0
        specified.append("tech_awareness")

    if financial_literacy_training is not None and "Fin_D4" in features:
        features["Fin_D4"] = 1.0 if financial_literacy_training else 0.0
        specified.append("financial_literacy_training")

    if feature_overrides:
        for k, v in feature_overrides.items():
            if k in features or k in feature_order:
                features[k] = float(v)
                specified.append(k)

    # Ensure every order key exists
    for col in feature_order:
        features.setdefault(col, 3.0 if is_likert_column(col) else 0.0)

    return features, specified


def features_dict_with_defaults(
    feature_order: Sequence[str],
    partial: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Merge partial full-feature dict onto defaults (no 422 for missing keys)."""
    base = build_default_features(feature_order)
    if partial:
        for k, v in partial.items():
            if v is None:
                continue
            try:
                base[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return {c: float(base.get(c, 0.0)) for c in feature_order}


def to_model_matrix(
    feature_order: Sequence[str], features: Dict[str, float]
) -> np.ndarray:
    """Row vector in feature_order for sklearn / MetaModel4."""
    return np.array(
        [float(features.get(c, 0.0)) for c in feature_order], dtype="float64"
    ).reshape(1, -1)


def to_model_frame(
    feature_order: Sequence[str], features: Dict[str, float]
) -> pd.DataFrame:
    return pd.DataFrame(
        [{c: float(features.get(c, 0.0)) for c in feature_order}]
    )
