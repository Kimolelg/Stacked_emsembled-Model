"""
Feast feature definitions — Teacher Mental Health Risk Prediction
================================================================

Aligned with this project's `src/data/processing.py` engineered feature matrix.

Entity
------
  teacher  — join key teacher_id (one row per survey respondent)

Feature views
-------------
  1. teacher_domain_fv
       Domain means (workload, WLB, financial, social, …) + key binaries.
       Small, stable schema for online serving & monitoring.

  2. teacher_survey_fv
       Full engineered feature vector (same names as feature_order.joblib).
       Schema is loaded from data/feast/feature_schema.json produced by
       `python -m src.data.feast_pipeline`. Ensures train/serve column parity.

Offline source
--------------
  data/feast/teacher_survey_features.parquet
  (built from raw TSC survey via processing.engineer_features_and_target)

Why this prevents training-serving skew
---------------------------------------
  Training (offline) and API (online) both read the same FeatureViews
  via FeatureStore.get_historical_features / get_online_features.
  Feature engineering still lives in processing.py; Feast is the
  serving contract over the *already engineered* matrix.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

REPO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPO_DIR.parent
FEAST_DATA = PROJECT_ROOT / "data" / "feast"
PARQUET_PATH = FEAST_DATA / "teacher_survey_features.parquet"
SCHEMA_PATH = FEAST_DATA / "feature_schema.json"

# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------
# join_keys is the modern Feast API (0.36+); value_type omitted for compatibility
teacher = Entity(
    name="teacher",
    join_keys=["teacher_id"],
    description="Teacher survey respondent (anonymized id)",
)
# ---------------------------------------------------------------------------
# Offline file source (wide table written by feast_pipeline)
# ---------------------------------------------------------------------------
teacher_features_source = FileSource(
    name="teacher_survey_features_source",
    path=str(PARQUET_PATH).replace("\\", "/"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ---------------------------------------------------------------------------
# Domain-level feature view (stable, small)
# ---------------------------------------------------------------------------
DOMAIN_FIELDS = [
    Field(name="workload_mean", dtype=Float32),
    Field(name="learners_mean", dtype=Float32),
    Field(name="wlb_mean", dtype=Float32),
    Field(name="perf_mean", dtype=Float32),
    Field(name="time_mean", dtype=Float32),
    Field(name="emot_mean", dtype=Float32),
    Field(name="fin_mean", dtype=Float32),
    Field(name="soc_mean", dtype=Float32),
    Field(name="tech_awareness", dtype=Float32),
    Field(name="fin_d4", dtype=Float32),
    Field(name="phq_total", dtype=Float32),  # for analysis only — not for prediction leakage
    Field(name="high_risk", dtype=Int64),  # label for offline joins; do not use as model feature
]

teacher_domain_fv = FeatureView(
    name="teacher_domain_fv",
    entities=[teacher],
    ttl=timedelta(days=3650),  # survey features are long-lived
    schema=DOMAIN_FIELDS,
    source=teacher_features_source,
    online=True,
    description=(
        "Domain aggregates from TSC survey (means of Likert blocks). "
        "phq_total/high_risk stored for offline labels only."
    ),
)


def _load_model_feature_names() -> list[str]:
    """Feature names for the full survey FeatureView (matches model feature_order)."""
    if SCHEMA_PATH.exists():
        data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        names = data.get("feature_order") or data.get("features") or []
        return [str(n) for n in names]
    # Fallback until first materialize — empty schema (apply still succeeds)
    return []


_model_feature_names = _load_model_feature_names()

# Feast field names must be valid identifiers — we keep engineered names as-is
# (underscores only after processing.clean_column_name).
_survey_fields = [Field(name=c, dtype=Float32) for c in _model_feature_names]

# Always define the view; if schema is empty, Feast apply still works and
# materialize will populate after feast_pipeline writes schema + parquet.
if _survey_fields:
    teacher_survey_fv = FeatureView(
        name="teacher_survey_fv",
        entities=[teacher],
        ttl=timedelta(days=3650),
        schema=_survey_fields,
        source=teacher_features_source,
        online=True,
        description=(
            "Full engineered survey feature vector — same columns as "
            "artifacts/feature_order.joblib used by Meta-Models 1–4."
        ),
    )
else:
    # Minimal placeholder so `feast apply` works before first pipeline run
    teacher_survey_fv = FeatureView(
        name="teacher_survey_fv",
        entities=[teacher],
        ttl=timedelta(days=3650),
        schema=[Field(name="workload_mean", dtype=Float32)],
        source=teacher_features_source,
        online=True,
        description="Placeholder until feast_pipeline generates full schema.",
    )
