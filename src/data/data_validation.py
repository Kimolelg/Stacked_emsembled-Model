"""
Data validation for Teacher Mental Health Risk Prediction
=========================================================

TSC survey / Meta-Models validation:

1. **Runtime gatekeeper** (fast custom Python) — every /predict request
2. **Batch auditor** (Great Expectations) — Feast parquet / training matrices

Philosophy: "Garbage in, error out" — reject bad input *before* the model runs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain constants (aligned with feature_builder / processing)
# ---------------------------------------------------------------------------

SECTION_FIELDS = {
    "workload",
    "learners",
    "work_life_balance",
    "performance",
    "time_pressure",
    "emotional_impact",
    "financial",
    "social_support",
}

# Acceptable *fragments* for demographics (fuzzy match after normalize)
VALID_AGE_FRAGMENTS = (
    "below25",
    "2534",
    "3544",
    "4554",
    "55",
)
VALID_GENDER_FRAGMENTS = ("female", "male", "other", "prefernot")
VALID_SCHOOL_FRAGMENTS = ("primary", "secondary", "junior")
VALID_YEARS_FRAGMENTS = (
    "lessthan5",
    "510",
    "1115",
    "1620",
    "over20",
    "morethan20",
)
VALID_EDU_FRAGMENTS = (
    "certificate",
    "diploma",
    "bachelor",
    "master",
    "phd",
    "degree",
)
VALID_ICT_FRAGMENTS = ("basic", "intermediate", "advanced", "expert")

LIKERT_PREFIXES = (
    "Workload_",
    "Learners_",
    "WLB_",
    "Perf_",
    "Time_",
    "Emot_",
    "Fin_",
    "Soc_",
)
BINARY_FEATURE_NAMES = {"Tech_Awareness", "Fin_D4"}


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower()) if s is not None else ""


def _matches_any(value: Any, fragments: Sequence[str]) -> bool:
    n = _norm(value)
    if not n:
        return False
    return any(f in n or n in f for f in fragments)


# ---------------------------------------------------------------------------
# Runtime validation (API)
# ---------------------------------------------------------------------------


def validate_questionnaire(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate compact questionnaire /predict body.

    Returns {"valid": bool, "errors": list[str], "warnings": list[str]}.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # teacher_id path
    tid = data.get("teacher_id")
    if tid is not None:
        try:
            tid_i = int(tid)
            if tid_i < 1:
                errors.append("teacher_id must be a positive integer (>= 1)")
        except (TypeError, ValueError):
            errors.append(f"teacher_id must be an integer (got {tid!r})")
        # When teacher_id is set, questionnaire fields are ignored — no further checks
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # At least one signal field should be present for a useful prediction
    signal_keys = list(SECTION_FIELDS) + [
        "age_category",
        "gender",
        "years_in_service",
        "school_type",
        "education_qualification",
        "ict_skills",
        "tech_awareness",
        "financial_literacy_training",
        "feature_overrides",
    ]
    provided = [k for k in signal_keys if data.get(k) is not None]
    if not provided:
        warnings.append(
            "No questionnaire fields provided; prediction will use all defaults "
            "(Likert=3, binary/OHE=0). Provide at least one section score or demographic."
        )

    # Section scores: 1–5 Likert averages
    for field in SECTION_FIELDS:
        val = data.get(field)
        if val is None:
            continue
        try:
            fval = float(val)
        except (TypeError, ValueError):
            errors.append(f"{field} must be a number between 1 and 5 (got {val!r})")
            continue
        if not (1.0 <= fval <= 5.0):
            errors.append(f"{field} must be between 1 and 5 inclusive (got {fval})")
        if fval != fval:  # NaN
            errors.append(f"{field} cannot be NaN")

    # Demographics (optional but constrained if present)
    demos = {
        "age_category": VALID_AGE_FRAGMENTS,
        "gender": VALID_GENDER_FRAGMENTS,
        "years_in_service": VALID_YEARS_FRAGMENTS,
        "school_type": VALID_SCHOOL_FRAGMENTS,
        "education_qualification": VALID_EDU_FRAGMENTS,
        "ict_skills": VALID_ICT_FRAGMENTS,
    }
    for field, frags in demos.items():
        val = data.get(field)
        if val is None or str(val).strip() == "":
            continue
        if not isinstance(val, str):
            errors.append(f"{field} must be a string (got {type(val).__name__})")
            continue
        if not _matches_any(val, frags):
            warnings.append(
                f"{field}={val!r} may not match training categories "
                f"(prediction will use defaults for that demographic)"
            )

    # Booleans
    for field in ("tech_awareness", "financial_literacy_training"):
        val = data.get(field)
        if val is None:
            continue
        if not isinstance(val, bool) and val not in (0, 1, "0", "1", "true", "false", "True", "False"):
            errors.append(f"{field} must be a boolean (got {val!r})")

    # feature_overrides
    overrides = data.get("feature_overrides")
    if overrides is not None:
        if not isinstance(overrides, dict):
            errors.append("feature_overrides must be an object/dict of feature→float")
        else:
            for k, v in overrides.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    errors.append(f"feature_overrides[{k!r}] must be numeric (got {v!r})")
                    continue
                if fv != fv:
                    errors.append(f"feature_overrides[{k!r}] cannot be NaN")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def validate_feature_vector(
    features: Dict[str, float],
    feature_order: Sequence[str],
    *,
    strict_keys: bool = False,
) -> Dict[str, Any]:
    """
    Validate a full (or expanded) engineered feature dict before model.predict.

    Checks:
    - required keys present (if strict_keys)
    - no NaN/Inf
    - Likert columns in [1, 5]
    - binary Tech_Awareness / Fin_D4 in {0, 1}
    - one-hot style columns roughly in [0, 1]
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not feature_order:
        errors.append("feature_order is empty — cannot validate against model schema")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if strict_keys:
        missing = [c for c in feature_order if c not in features]
        if missing:
            errors.append(
                f"Missing {len(missing)} features required by model "
                f"(e.g. {missing[:5]})"
            )

    n_nan = 0
    n_likert_bad = 0
    n_binary_bad = 0

    for col in feature_order:
        if col not in features:
            continue
        try:
            v = float(features[col])
        except (TypeError, ValueError):
            errors.append(f"{col} must be numeric (got {features[col]!r})")
            continue

        if not np.isfinite(v):
            n_nan += 1
            continue

        # Likert domains
        if any(col.startswith(p) for p in LIKERT_PREFIXES) and col not in BINARY_FEATURE_NAMES:
            if not (1.0 <= v <= 5.0):
                # Allow 0 from incomplete OHE-like bugs but flag
                if v == 0.0:
                    warnings.append(f"{col}={v} is outside Likert 1–5 (treating as missing-like)")
                else:
                    n_likert_bad += 1
        elif col in BINARY_FEATURE_NAMES or col.startswith("Tech_Platforms_"):
            if v not in (0.0, 1.0) and not (0.0 <= v <= 1.0):
                n_binary_bad += 1
        elif any(
            col.startswith(p)
            for p in (
                "Age_Category_",
                "Gender_",
                "Years_in_Service_",
                "School_Type_",
                "Education_Qualification_",
                "ICT_Skills_",
                "Sub_County_",
            )
        ):
            if not (0.0 <= v <= 1.0):
                warnings.append(f"{col}={v} outside [0,1] for one-hot-like column")

    if n_nan:
        errors.append(f"{n_nan} feature value(s) are NaN or Inf")
    if n_likert_bad:
        errors.append(
            f"{n_likert_bad} Likert feature(s) outside range 1–5 "
            "(check section scores / overrides)"
        )
    if n_binary_bad:
        errors.append(f"{n_binary_bad} binary feature(s) outside {{0,1}}")

    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def validate_prediction_request(
    *,
    questionnaire: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, float]] = None,
    feature_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Combined gate for API: questionnaire and/or engineered features.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if questionnaire is not None:
        q = validate_questionnaire(questionnaire)
        errors.extend(q["errors"])
        warnings.extend(q.get("warnings") or [])

    if features is not None and feature_order is not None:
        f = validate_feature_vector(features, feature_order)
        errors.extend(f["errors"])
        warnings.extend(f.get("warnings") or [])

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Batch validation with Great Expectations (training / Feast audits)
# ---------------------------------------------------------------------------


def _ge_dataset(df: pd.DataFrame):
    """
    Build a GE-compatible dataset across GX versions.

    Prefer gx.from_pandas(df) on GE 0.18.
    Newer GX may need PandasDataset or ephemeral context.
    """
    try:
        import great_expectations as gx

        if hasattr(gx, "from_pandas"):
            return gx.from_pandas(df), "gx.from_pandas"
    except Exception as exc:  # noqa: BLE001
        logger.debug("gx.from_pandas unavailable: %s", exc)

    try:
        from great_expectations.dataset import PandasDataset

        return PandasDataset(df), "PandasDataset"
    except Exception as exc:  # noqa: BLE001
        logger.debug("PandasDataset unavailable: %s", exc)

    return None, None


def _run_expect(ge_df, method: str, *args, **kwargs) -> Dict[str, Any]:
    fn = getattr(ge_df, method, None)
    if fn is None:
        return {"success": False, "skipped": True, "detail": f"{method} not available"}
    try:
        r = fn(*args, **kwargs)
        # GE returns ExpectationValidationResult with .success
        success = bool(getattr(r, "success", r))
        result = getattr(r, "result", {}) or {}
        return {"success": success, "skipped": False, "detail": result}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "skipped": False, "detail": {"exception": str(exc)}}


def validate_batch_features(
    df: pd.DataFrame,
    feature_order: Optional[Sequence[str]] = None,
    *,
    include_label: bool = True,
) -> Dict[str, Any]:
    """
    Validate a batch DataFrame (Feast offline table or training matrix)
    with Great Expectations when available.

    Expectations (mental-health specific):
    - No nulls in model feature columns (mostly)
    - Likert-like columns between 1 and 5 (mostly ≥ 0.95)
    - Binary columns in {0, 1}
    - high_risk in {0, 1} if present
    - phq_total between 0 and 27 if present
    - teacher_id unique & positive if present
    """
    if df is None or len(df) == 0:
        return {
            "success": False,
            "passed": 0,
            "total": 0,
            "pass_rate": 0.0,
            "errors": ["Empty DataFrame"],
            "details": {},
            "engine": "none",
        }

    work = df.copy()
    # Infer feature columns
    if feature_order:
        cols = [c for c in feature_order if c in work.columns]
    else:
        cols = [
            c
            for c in work.columns
            if c
            not in (
                "teacher_id",
                "event_timestamp",
                "created_timestamp",
                "high_risk",
                "phq_total",
                "Depression_Severity",
                "PHQ_Total",
            )
        ]

    ge_df, engine = _ge_dataset(work)
    details: Dict[str, Dict[str, Any]] = {}
    results_flags: List[bool] = []

    if ge_df is not None:
        # --- GE expectations ---
        # Row count
        r = _run_expect(ge_df, "expect_table_row_count_to_be_between", min_value=1)
        details["table_row_count"] = r
        if not r.get("skipped"):
            results_flags.append(r["success"])

        # teacher_id
        if "teacher_id" in work.columns:
            r = _run_expect(ge_df, "expect_column_values_to_not_be_null", "teacher_id")
            details["teacher_id_not_null"] = r
            results_flags.append(r["success"])
            r = _run_expect(
                ge_df, "expect_column_values_to_be_unique", "teacher_id"
            )
            details["teacher_id_unique"] = r
            results_flags.append(r["success"])
            r = _run_expect(
                ge_df,
                "expect_column_values_to_be_between",
                "teacher_id",
                min_value=1,
                max_value=10_000_000,
            )
            details["teacher_id_positive"] = r
            results_flags.append(r["success"])

        # Label / PHQ
        if include_label and "high_risk" in work.columns:
            r = _run_expect(
                ge_df, "expect_column_values_to_be_in_set", "high_risk", [0, 1]
            )
            details["high_risk_binary"] = r
            results_flags.append(r["success"])
        if "phq_total" in work.columns:
            r = _run_expect(
                ge_df,
                "expect_column_values_to_be_between",
                "phq_total",
                min_value=0,
                max_value=27,
                mostly=0.99,
            )
            details["phq_total_range"] = r
            results_flags.append(r["success"])

        # Sample of Likert columns (all would be slow / noisy; check by prefix)
        likert_cols = [
            c
            for c in cols
            if any(c.startswith(p) for p in LIKERT_PREFIXES)
            and c not in BINARY_FEATURE_NAMES
        ][:40]  # cap for speed
        for c in likert_cols:
            r = _run_expect(
                ge_df,
                "expect_column_values_to_be_between",
                c,
                min_value=1,
                max_value=5,
                mostly=0.95,
            )
            details[f"{c}_likert_range"] = r
            results_flags.append(r["success"])

        for c in BINARY_FEATURE_NAMES:
            if c in work.columns:
                r = _run_expect(
                    ge_df, "expect_column_values_to_be_in_set", c, [0, 1, 0.0, 1.0]
                )
                details[f"{c}_binary"] = r
                results_flags.append(r["success"])

        # Null rate on a sample of model features
        for c in cols[:30]:
            r = _run_expect(
                ge_df, "expect_column_values_to_not_be_null", c, mostly=0.99
            )
            details[f"{c}_not_null"] = r
            results_flags.append(r["success"])

    else:
        # --- Pure pandas fallback (same rules, no GE package) ---
        engine = "pandas_fallback"
        if "teacher_id" in work.columns:
            ok = work["teacher_id"].notna().all() and work["teacher_id"].is_unique
            details["teacher_id_basic"] = {"success": bool(ok)}
            results_flags.append(bool(ok))
        if "high_risk" in work.columns:
            ok = work["high_risk"].isin([0, 1]).all()
            details["high_risk_binary"] = {"success": bool(ok)}
            results_flags.append(bool(ok))
        if "phq_total" in work.columns:
            ok = work["phq_total"].between(0, 27).mean() >= 0.99
            details["phq_total_range"] = {"success": bool(ok)}
            results_flags.append(bool(ok))
        likert_cols = [
            c
            for c in cols
            if any(c.startswith(p) for p in LIKERT_PREFIXES)
            and c not in BINARY_FEATURE_NAMES
        ][:40]
        for c in likert_cols:
            ok = work[c].between(1, 5).mean() >= 0.95
            details[f"{c}_likert_range"] = {"success": bool(ok)}
            results_flags.append(bool(ok))

    passed = sum(1 for x in results_flags if x)
    total = len(results_flags)
    success = total > 0 and passed == total

    return {
        "success": success,
        "passed": passed,
        "total": total,
        "pass_rate": (passed / total) if total else 0.0,
        "details": {
            k: {
                "passed": v.get("success", False),
                "skipped": v.get("skipped", False),
            }
            for k, v in details.items()
        },
        "engine": engine or "none",
        "n_rows": int(len(work)),
        "n_feature_cols_checked": len(cols),
    }


def validate_feast_parquet(
    parquet_path: Optional[Union[str, Path]] = None,
    feature_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Validate the Feast offline table on disk."""
    root = Path(__file__).resolve().parents[2]
    path = Path(parquet_path or root / "data" / "feast" / "teacher_survey_features.parquet")
    if not path.exists():
        return {
            "success": False,
            "passed": 0,
            "total": 0,
            "pass_rate": 0.0,
            "errors": [f"Parquet not found: {path}"],
            "details": {},
            "engine": "none",
        }
    df = pd.read_parquet(path)
    if feature_order is None:
        schema = root / "data" / "feast" / "feature_schema.json"
        if schema.exists():
            import json

            feature_order = json.loads(schema.read_text(encoding="utf-8")).get(
                "feature_order"
            )
    return validate_batch_features(df, feature_order=feature_order)


def raise_http_validation_error(validation: Dict[str, Any], input_snapshot: Any = None):
    """Raise FastAPI HTTPException 400 for failed validation."""
    from fastapi import HTTPException

    raise HTTPException(
        status_code=400,
        detail={
            "message": "Validation failed",
            "errors": validation.get("errors") or [],
            "warnings": validation.get("warnings") or [],
            "input": input_snapshot,
        },
    )
