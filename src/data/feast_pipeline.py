"""
Feast materialization pipeline for Teacher Mental Health
========================================================

TSC survey + processing.py:

  1. Load raw survey Excel
  2. engineer_features_and_target + prepare_for_modeling
  3. Build wide Feast table (teacher_id, timestamps, domain means, full features)
  4. Write data/feast/teacher_survey_features.parquet + feature_schema.json
  5. feast apply + materialize into offline/online stores

Usage:
  python -m src.data.feast_pipeline --data_path path/to/survey.xlsx
  python -m src.data.feast_pipeline --data_path path/to/survey.xlsx --skip-materialize
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.processing import (  # noqa: E402
    engineer_features_and_target,
    load_and_clean_raw_data,
    prepare_for_modeling,
)

logger = logging.getLogger(__name__)

FEAST_DATA = PROJECT_ROOT / "data" / "feast"
FEATURE_REPO = PROJECT_ROOT / "feature_repo"
PARQUET_PATH = FEAST_DATA / "teacher_survey_features.parquet"
SCHEMA_PATH = FEAST_DATA / "feature_schema.json"
LABELS_PATH = FEAST_DATA / "teacher_labels.parquet"
ARTIFACTS = PROJECT_ROOT / "artifacts"

# Prefixes for domain means (match engineered short names)
DOMAIN_PREFIXES: Dict[str, str] = {
    "workload_mean": "Workload_",
    "learners_mean": "Learners_",
    "wlb_mean": "WLB_",
    "perf_mean": "Perf_",
    "time_mean": "Time_",
    "emot_mean": "Emot_",
    "fin_mean": "Fin_",
    "soc_mean": "Soc_",
}


def _domain_means(X: pd.DataFrame) -> pd.DataFrame:
    """Compute mean Likert score per survey domain (exclude binary Fin_D4)."""
    out = pd.DataFrame(index=X.index)
    for col_name, prefix in DOMAIN_PREFIXES.items():
        cols = [
            c
            for c in X.columns
            if c.startswith(prefix) and c not in ("Fin_D4",)
        ]
        if cols:
            out[col_name] = X[cols].astype(float).mean(axis=1)
        else:
            out[col_name] = 3.0  # neutral default
    return out


def build_feast_table(
    data_path: str,
    sheet_name: str = "Sheet1",
) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Run processing pipeline and assemble Feast offline table.

    Returns
    -------
    feast_df : wide table with teacher_id, timestamps, domain + model features
    feature_order : model feature columns (excludes labels/ids)
    y : High_Risk labels aligned to feast_df index order
    """
    logger.info("Loading & engineering survey: %s", data_path)
    df_raw = load_and_clean_raw_data(data_path, sheet_name=sheet_name)
    df_eng = engineer_features_and_target(df_raw)
    X, y, feature_order = prepare_for_modeling(df_eng)
    if y is None:
        raise ValueError("High_Risk target missing after engineering.")

    n = len(X)
    # Synthetic teacher ids (anonymized row index); stable for re-runs with same data order
    teacher_ids = np.arange(1, n + 1, dtype=np.int64)

    # Point-in-time timestamps: staggered slightly so historical retrieval is valid
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    event_ts = [now - timedelta(days=1) for _ in range(n)]
    created_ts = [now for _ in range(n)]

    domain = _domain_means(X)
    feast_df = pd.DataFrame(
        {
            "teacher_id": teacher_ids,
            "event_timestamp": event_ts,
            "created_timestamp": created_ts,
        }
    )
    feast_df = pd.concat(
        [feast_df, domain.reset_index(drop=True), X.reset_index(drop=True)],
        axis=1,
    )

    # Label columns for offline analysis (stored in parquet; domain FV exposes them)
    feast_df["tech_awareness"] = (
        X["Tech_Awareness"].astype(float).values
        if "Tech_Awareness" in X.columns
        else 0.0
    )
    feast_df["fin_d4"] = (
        X["Fin_D4"].astype(float).values if "Fin_D4" in X.columns else 0.0
    )
    phq = (
        df_eng.loc[X.index, "PHQ_Total"].astype(float).values
        if "PHQ_Total" in df_eng.columns
        else np.zeros(n)
    )
    feast_df["phq_total"] = phq
    feast_df["high_risk"] = y.astype(np.int64).values

    # Ensure float32 for model feature columns (Feast Float32)
    for c in feature_order:
        if c in feast_df.columns:
            feast_df[c] = feast_df[c].astype("float32")

    for c in DOMAIN_PREFIXES:
        feast_df[c] = feast_df[c].astype("float32")

    return feast_df, feature_order, y.reset_index(drop=True)


def write_feast_artifacts(
    feast_df: pd.DataFrame,
    feature_order: Sequence[str],
) -> Dict[str, Any]:
    """Write parquet + schema JSON + labels; sync feature_order.joblib."""
    FEAST_DATA.mkdir(parents=True, exist_ok=True)

    # Parquet needs timezone-naive or consistent timestamps
    out = feast_df.copy()
    for col in ("event_timestamp", "created_timestamp"):
        out[col] = pd.to_datetime(out[col])

    out.to_parquet(PARQUET_PATH, index=False)
    logger.info("Wrote %s shape=%s", PARQUET_PATH, out.shape)

    schema = {
        "feature_order": list(feature_order),
        "n_features": len(feature_order),
        "domain_features": list(DOMAIN_PREFIXES.keys())
        + ["tech_awareness", "fin_d4", "phq_total", "high_risk"],
        "entity": "teacher_id",
        "n_rows": int(len(out)),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source": "src.data.processing + feast_pipeline",
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    logger.info("Wrote schema (%s model features)", len(feature_order))

    labels = out[["teacher_id", "event_timestamp", "high_risk", "phq_total"]].copy()
    labels.to_parquet(LABELS_PATH, index=False)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(list(feature_order), ARTIFACTS / "feature_order.joblib")
    logger.info("Synced artifacts/feature_order.joblib")

    return schema


def _feast_cli(*args: str) -> subprocess.CompletedProcess:
    """
    Invoke the Feast CLI.

    Note: `python -m feast` does NOT work (no feast.__main__).
    Prefer the console script `feast` next to the current interpreter.
    """
    scripts_dir = Path(sys.executable).resolve().parent
    candidates = [
        scripts_dir / "feast.exe",
        scripts_dir / "feast",
        "feast",  # PATH fallback
    ]
    last_err: Optional[Exception] = None
    for cmd in candidates:
        try:
            if isinstance(cmd, Path) and not cmd.exists():
                continue
            return subprocess.run(
                [str(cmd), *args],
                cwd=str(FEATURE_REPO),
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(
        f"Could not run Feast CLI (tried {candidates}). Last error: {last_err}"
    )


def feast_apply_and_materialize(
    end_date: Optional[datetime] = None,
) -> None:
    """Run `feast apply` then materialize offline → online."""
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PARQUET_PATH}. Run feast_pipeline with survey data first."
        )
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Missing {SCHEMA_PATH}")

    # feast apply must re-import features.py with the fresh schema JSON
    logger.info("Running feast apply in %s ...", FEATURE_REPO)
    proc = _feast_cli("apply")
    if proc.returncode != 0:
        logger.error("feast apply stdout:\n%s", proc.stdout)
        logger.error("feast apply stderr:\n%s", proc.stderr)
        raise RuntimeError(
            f"feast apply failed (code {proc.returncode}). "
            "Ensure feast is installed in this venv: pip install feast==0.40.1"
        )
    logger.info("feast apply OK\n%s", (proc.stdout or "")[-2000:])

    from feast import FeatureStore

    store = FeatureStore(repo_path=str(FEATURE_REPO))
    end = end_date or datetime.utcnow()
    # Wide window so all survey event_timestamps fall inside the materialize range
    start = end - timedelta(days=365)
    logger.info("Materializing %s → %s ...", start.isoformat(), end.isoformat())
    try:
        store.materialize(start_date=start, end_date=end)
    except TypeError:
        # Some Feast versions use different kw names
        store.materialize(start, end)
    logger.info("Materialize complete.")


def run_pipeline(
    data_path: str,
    sheet_name: str = "Sheet1",
    skip_materialize: bool = False,
) -> Dict[str, Any]:
    """Full Feast pipeline from raw survey."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    feast_df, feature_order, y = build_feast_table(data_path, sheet_name=sheet_name)
    schema = write_feast_artifacts(feast_df, feature_order)

    result = {
        "n_teachers": int(len(feast_df)),
        "n_features": len(feature_order),
        "high_risk_rate": float(y.mean()),
        "parquet": str(PARQUET_PATH),
        "schema": str(SCHEMA_PATH),
        "materialized": False,
    }

    # Batch data-quality audit (Great Expectations when installed)
    try:
        from src.data.data_validation import validate_batch_features

        batch = validate_batch_features(
            feast_df, feature_order=feature_order, include_label=True
        )
        result["validation"] = {
            "success": batch.get("success"),
            "passed": batch.get("passed"),
            "total": batch.get("total"),
            "pass_rate": batch.get("pass_rate"),
            "engine": batch.get("engine"),
        }
        if batch.get("success"):
            logger.info(
                "Batch validation OK (%s/%s, engine=%s)",
                batch.get("passed"),
                batch.get("total"),
                batch.get("engine"),
            )
        else:
            logger.warning(
                "Batch validation issues: %s/%s passed (engine=%s)",
                batch.get("passed"),
                batch.get("total"),
                batch.get("engine"),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Batch validation skipped: %s", exc)
        result["validation_error"] = str(exc)

    if not skip_materialize:
        try:
            feast_apply_and_materialize()
            result["materialized"] = True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Materialize failed (parquet still written): %s", exc)
            result["materialize_error"] = str(exc)
    else:
        logger.info("Skipped feast apply/materialize (--skip-materialize)")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Feast offline table from TSC survey and materialize"
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sheet_name", type=str, default="Sheet1")
    parser.add_argument(
        "--skip-materialize",
        action="store_true",
        help="Only write parquet/schema (no feast apply)",
    )
    args = parser.parse_args()
    info = run_pipeline(
        data_path=args.data_path,
        sheet_name=args.sheet_name,
        skip_materialize=args.skip_materialize,
    )
    print(json.dumps(info, indent=2))
