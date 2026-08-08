"""
Feast client helpers — offline training matrix + online serving vectors
======================================================================

Training (offline):
  get_training_matrix() → X, y, feature_order, entity_df
  Uses FeatureStore.get_historical_features when Feast is applied;
  falls back to reading the parquet directly if Feast is unavailable.

Serving (online):
  get_online_feature_row(teacher_id) → dict aligned to feature_order
  Falls back to parquet lookup if online store empty / Feast missing.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

FEAST_DATA = PROJECT_ROOT / "data" / "feast"
FEATURE_REPO = PROJECT_ROOT / "feature_repo"
PARQUET_PATH = FEAST_DATA / "teacher_survey_features.parquet"
SCHEMA_PATH = FEAST_DATA / "feature_schema.json"
LABELS_PATH = FEAST_DATA / "teacher_labels.parquet"


def load_feature_order() -> List[str]:
    if SCHEMA_PATH.exists():
        data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        return list(data.get("feature_order") or [])
    art = PROJECT_ROOT / "artifacts" / "feature_order.joblib"
    if art.exists():
        import joblib

        return list(joblib.load(art))
    raise FileNotFoundError(
        "No feature_order. Run: python -m src.data.feast_pipeline --data_path <survey>"
    )


def _get_store():
    from feast import FeatureStore

    return FeatureStore(repo_path=str(FEATURE_REPO))


def get_training_matrix(
    use_feast_historical: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Offline feature matrix for model training.

    Returns
    -------
    X : DataFrame of model features (feature_order columns)
    y : High_Risk series
    feature_order : list[str]
    entity_df : teacher_id + event_timestamp (for lineage / MLflow)
    """
    feature_order = load_feature_order()
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PARQUET_PATH}. Materialize Feast features first."
        )

    raw = pd.read_parquet(PARQUET_PATH)
    entity_df = raw[["teacher_id", "event_timestamp"]].copy()
    y = raw["high_risk"].astype(int)

    if use_feast_historical:
        try:
            store = _get_store()
            feature_refs = [f"teacher_survey_fv:{c}" for c in feature_order]
            # Only request features that exist in the view
            hist = store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs,
            ).to_df()
            # Columns may be prefixed; normalize
            X = pd.DataFrame(index=hist.index)
            for c in feature_order:
                if c in hist.columns:
                    X[c] = hist[c].astype(float)
                else:
                    # Feast sometimes returns feature view prefix
                    alt = f"teacher_survey_fv__{c}"
                    if alt in hist.columns:
                        X[c] = hist[alt].astype(float)
                    else:
                        X[c] = 0.0
            # Align y to historical output order via teacher_id merge
            if "teacher_id" in hist.columns:
                merged = hist[["teacher_id"]].merge(
                    raw[["teacher_id", "high_risk"]],
                    on="teacher_id",
                    how="left",
                )
                y = merged["high_risk"].astype(int)
                entity_df = hist[["teacher_id", "event_timestamp"]].copy()
            logger.info(
                "Offline features via Feast historical API: %s", X.shape
            )
            return X[feature_order], y, feature_order, entity_df
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Feast historical retrieval failed (%s); using parquet fallback",
                exc,
            )

    X = raw[feature_order].astype(float)
    logger.info("Offline features via parquet: %s", X.shape)
    return X, y, feature_order, entity_df


def get_online_feature_row(
    teacher_id: int,
    feature_order: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Online feature vector for one teacher (serving).

    Tries Feast online store first; falls back to parquet row lookup.
    """
    feature_order = list(feature_order or load_feature_order())

    try:
        store = _get_store()
        feature_refs = [f"teacher_survey_fv:{c}" for c in feature_order]
        online = store.get_online_features(
            features=feature_refs,
            entity_rows=[{"teacher_id": int(teacher_id)}],
        ).to_dict()
        row: Dict[str, float] = {}
        for c in feature_order:
            vals = online.get(c) or online.get(f"teacher_survey_fv__{c}")
            if vals is None:
                row[c] = 0.0
            else:
                v = vals[0]
                row[c] = float(v) if v is not None else 0.0
        # If all zeros and teacher missing, fall through to parquet
        if any(v != 0.0 for v in row.values()):
            logger.info("Online features for teacher_id=%s via Feast", teacher_id)
            return row
    except Exception as exc:  # noqa: BLE001
        logger.warning("Feast online get failed (%s); parquet fallback", exc)

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"teacher_id={teacher_id} not found and Feast/parquet unavailable"
        )
    raw = pd.read_parquet(PARQUET_PATH)
    hit = raw.loc[raw["teacher_id"] == int(teacher_id)]
    if hit.empty:
        raise KeyError(f"teacher_id={teacher_id} not in Feast offline table")
    r = hit.iloc[0]
    return {c: float(r[c]) if c in r.index else 0.0 for c in feature_order}


def get_domain_features_online(teacher_id: int) -> Dict[str, Any]:
    """Fetch domain aggregates for monitoring / simplified display."""
    domain_cols = [
        "workload_mean",
        "learners_mean",
        "wlb_mean",
        "perf_mean",
        "time_mean",
        "emot_mean",
        "fin_mean",
        "soc_mean",
        "tech_awareness",
        "fin_d4",
    ]
    try:
        store = _get_store()
        refs = [f"teacher_domain_fv:{c}" for c in domain_cols]
        online = store.get_online_features(
            features=refs,
            entity_rows=[{"teacher_id": int(teacher_id)}],
        ).to_dict()
        return {c: (online.get(c) or [None])[0] for c in domain_cols}
    except Exception:  # noqa: BLE001
        raw = pd.read_parquet(PARQUET_PATH)
        hit = raw.loc[raw["teacher_id"] == int(teacher_id)]
        if hit.empty:
            return {}
        r = hit.iloc[0]
        return {c: float(r[c]) if c in r.index else None for c in domain_cols}


def list_teacher_ids(limit: int = 20) -> List[int]:
    """Sample teacher_ids present in the Feast offline table."""
    if not PARQUET_PATH.exists():
        return []
    raw = pd.read_parquet(PARQUET_PATH, columns=["teacher_id"])
    return [int(x) for x in raw["teacher_id"].head(limit).tolist()]
