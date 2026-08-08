"""
Evidently drift monitoring — Teacher Mental Health Risk
=======================================================

Reference data  = training / Feast offline survey features (baseline world)
  Current data    = production prediction log (live API traffic)

Detects:
  - Data drift on domain feature means (workload, WLB, financial, social, …)
  - Prediction drift (prediction_proba / prediction class distribution)

Also provides a scipy KS-test fallback if Evidently is unavailable.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITOR_DIR = PROJECT_ROOT / "data" / "monitoring"
REPORTS_DIR = MONITOR_DIR / "reports"
REFERENCE_PATH = MONITOR_DIR / "reference_snapshot.parquet"
SUMMARY_PATH = MONITOR_DIR / "last_drift_summary.json"

# Columns used for drift (stable domain features + predictions)
DRIFT_FEATURE_COLS = [
    "workload_mean",
    "learners_mean",
    "wlb_mean",
    "perf_mean",
    "time_mean",
    "emot_mean",
    "fin_mean",
    "soc_mean",
    "tech_awareness",
]
PRED_COLS = ["prediction_proba", "prediction"]


def _domain_means_from_matrix(X: pd.DataFrame) -> pd.DataFrame:
    prefixes = {
        "workload_mean": "Workload_",
        "learners_mean": "Learners_",
        "wlb_mean": "WLB_",
        "perf_mean": "Perf_",
        "time_mean": "Time_",
        "emot_mean": "Emot_",
        "fin_mean": "Fin_",
        "soc_mean": "Soc_",
    }
    out = pd.DataFrame(index=X.index)
    for name, prefix in prefixes.items():
        cols = [c for c in X.columns if c.startswith(prefix) and c != "Fin_D4"]
        out[name] = X[cols].astype(float).mean(axis=1) if cols else 3.0
    out["tech_awareness"] = (
        X["Tech_Awareness"].astype(float) if "Tech_Awareness" in X.columns else 0.0
    )
    return out


def build_reference_from_feast(
    max_rows: int = 5000,
    model: Any = None,
    feature_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build reference snapshot from Feast offline table (preferred) or fail clearly.
    Optionally scores rows with `model` to include prediction columns.
    """
    feast_parquet = PROJECT_ROOT / "data" / "feast" / "teacher_survey_features.parquet"
    if not feast_parquet.exists():
        raise FileNotFoundError(
            f"Missing {feast_parquet}. Run: python -m src.data.feast_pipeline --data_path <survey>"
        )

    raw = pd.read_parquet(feast_parquet)
    if len(raw) > max_rows:
        raw = raw.sample(n=max_rows, random_state=42).reset_index(drop=True)

    # Domain means already in feast table if pipeline wrote them
    ref = pd.DataFrame()
    for c in DRIFT_FEATURE_COLS:
        if c in raw.columns:
            ref[c] = raw[c].astype(float)
        else:
            # recompute from engineered cols
            eng = _domain_means_from_matrix(raw)
            ref = eng[DRIFT_FEATURE_COLS].copy()
            break
    else:
        if ref.empty:
            ref = _domain_means_from_matrix(raw)

    if "high_risk" in raw.columns:
        ref["target"] = raw["high_risk"].astype(int).values

    # Optional model scores for prediction-drift baseline
    if model is not None and feature_order is not None:
        cols = [c for c in feature_order if c in raw.columns]
        X = raw[cols].astype(float)
        # pad missing
        for c in feature_order:
            if c not in X.columns:
                X[c] = 0.0
        X = X[list(feature_order)]
        try:
            proba = model.predict_proba(X)[:, 1]
            pred = model.predict(X)
            ref["prediction_proba"] = proba
            ref["prediction"] = pred.astype(int)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not score reference with model: %s", exc)

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(REFERENCE_PATH, index=False)
    logger.info("Wrote reference snapshot %s shape=%s", REFERENCE_PATH, ref.shape)
    return ref


def load_reference() -> pd.DataFrame:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"No reference snapshot at {REFERENCE_PATH}. "
            "Call POST /monitoring/build-reference or run scripts/run_drift_check."
        )
    return pd.read_parquet(REFERENCE_PATH)


def _ks_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str],
    p_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Lightweight KS-test fallback when Evidently is unavailable."""
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        return {
            "engine": "ks_unavailable",
            "dataset_drift": False,
            "share_drifted": 0.0,
            "columns": {},
        }

    col_results = {}
    drifted = 0
    checked = 0
    for c in columns:
        if c not in reference.columns or c not in current.columns:
            continue
        a = reference[c].dropna().astype(float)
        b = current[c].dropna().astype(float)
        if len(a) < 5 or len(b) < 5:
            continue
        stat, pval = ks_2samp(a, b)
        is_drift = bool(pval < p_threshold)
        col_results[c] = {
            "statistic": float(stat),
            "p_value": float(pval),
            "drift_detected": is_drift,
        }
        checked += 1
        if is_drift:
            drifted += 1

    share = (drifted / checked) if checked else 0.0
    return {
        "engine": "scipy_ks",
        "dataset_drift": share >= 0.3,  # ≥30% of columns drifted
        "share_drifted": share,
        "n_drifted": drifted,
        "n_checked": checked,
        "columns": col_results,
    }


def _evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str],
    report_path: Path,
) -> Dict[str, Any]:
    """
    Generate Evidently HTML report. Supports both older (0.4/0.7) and newer APIs.
    """
    ref = reference[[c for c in columns if c in reference.columns]].copy()
    cur = current[[c for c in columns if c in current.columns]].copy()
    # Align columns
    common = [c for c in columns if c in ref.columns and c in cur.columns]
    ref = ref[common]
    cur = cur[common]

    # --- Try classic Evidently API (0.4.x) ---
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        metrics = [DataDriftPreset()]
        # Target / prediction drift if columns present
        try:
            from evidently.metric_preset import TargetDriftPreset

            if "target" in reference.columns and "target" in current.columns:
                ref["target"] = reference["target"]
                cur["target"] = current["target"]
                metrics.append(TargetDriftPreset())
        except Exception:  # noqa: BLE001
            pass

        if "prediction" in reference.columns and "prediction" in current.columns:
            # Treat prediction as classification label column for drift
            ref["prediction"] = reference["prediction"]
            cur["prediction"] = current["prediction"]

        if (
            "prediction_proba" in reference.columns
            and "prediction_proba" in current.columns
        ):
            ref["prediction_proba"] = reference["prediction_proba"]
            cur["prediction_proba"] = current["prediction_proba"]

        report = Report(metrics=metrics)
        report.run(reference_data=ref, current_data=cur)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(report_path))

        # Extract summary if possible
        dataset_drift = None
        share = None
        try:
            as_dict = report.as_dict()
            for m in as_dict.get("metrics", []):
                res = m.get("result") or {}
                if "dataset_drift" in res:
                    dataset_drift = res["dataset_drift"]
                if "share_of_drifted_columns" in res:
                    share = res["share_of_drifted_columns"]
        except Exception:  # noqa: BLE001
            pass

        return {
            "engine": "evidently_report",
            "report_path": str(report_path),
            "dataset_drift": dataset_drift,
            "share_drifted": share,
            "columns_used": common,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Classic Evidently Report API failed: %s", exc)

    # --- Newer Evidently API (0.5+ / 0.6+) ---
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        result = report.run(reference_data=ref, current_data=cur)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        # save html
        if hasattr(result, "save_html"):
            result.save_html(str(report_path))
        elif hasattr(report, "save_html"):
            report.save_html(str(report_path))
        else:
            # try HTML string
            html = None
            if hasattr(result, "html"):
                html = result.html()
            if html:
                report_path.write_text(html, encoding="utf-8")

        return {
            "engine": "evidently_presets",
            "report_path": str(report_path),
            "dataset_drift": None,
            "share_drifted": None,
            "columns_used": common,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("New Evidently API failed: %s", exc)
        raise


class DriftMonitor:
    """
    Compare reference (training/Feast) vs current (production log).

    Usage:
        mon = DriftMonitor()
        mon.set_reference_from_feast()
        summary = mon.check_drift()
    """

    def __init__(
        self,
        drift_share_threshold: float = 0.3,
        min_current_rows: int = 30,
    ):
        self.drift_share_threshold = drift_share_threshold
        self.min_current_rows = min_current_rows
        MONITOR_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def set_reference_from_feast(
        self,
        max_rows: int = 5000,
        model: Any = None,
        feature_order: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        ref = build_reference_from_feast(
            max_rows=max_rows, model=model, feature_order=feature_order
        )
        return {"n_rows": len(ref), "path": str(REFERENCE_PATH), "columns": list(ref.columns)}

    def check_drift(
        self,
        current: Optional[pd.DataFrame] = None,
        reference: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run drift analysis. Returns summary dict and writes HTML + JSON.
        """
        from .prediction_logger import get_prediction_logger

        ref = reference if reference is not None else load_reference()
        if current is None:
            current = get_prediction_logger().load_production()

        if current is None or len(current) < self.min_current_rows:
            return {
                "success": False,
                "reason": (
                    f"Need at least {self.min_current_rows} production predictions "
                    f"(have {0 if current is None else len(current)}). "
                    "Call /predict a few times, then re-run drift check."
                ),
                "n_reference": len(ref),
                "n_current": 0 if current is None else len(current),
                "dataset_drift": False,
                "alert": False,
            }

        # Align feature columns for KS + Evidently
        feature_cols = [c for c in DRIFT_FEATURE_COLS if c in ref.columns and c in current.columns]
        # Include prediction cols in report if present on both
        report_cols = list(feature_cols)
        for c in PRED_COLS:
            if c in ref.columns and c in current.columns:
                report_cols.append(c)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"drift_report_{stamp}.html"
        latest_path = REPORTS_DIR / "latest_drift_report.html"

        # KS fallback always computed
        ks = _ks_drift(ref, current, feature_cols)

        evid: Dict[str, Any] = {}
        try:
            evid = _evidently_report(ref, current, report_cols, report_path)
            # copy as latest
            if report_path.exists():
                latest_path.write_bytes(report_path.read_bytes())
                evid["latest_report_path"] = str(latest_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evidently report generation failed: %s", exc)
            evid = {
                "engine": "evidently_failed",
                "error": str(exc),
                "report_path": None,
            }

        # Decide alert
        share = evid.get("share_drifted")
        if share is None:
            share = ks.get("share_drifted", 0.0)
        dataset_drift = evid.get("dataset_drift")
        if dataset_drift is None:
            dataset_drift = ks.get("dataset_drift", False)

        alert = bool(dataset_drift) or (
            share is not None and float(share) >= self.drift_share_threshold
        )

        summary = {
            "success": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_reference": int(len(ref)),
            "n_current": int(len(current)),
            "feature_columns": feature_cols,
            "dataset_drift": bool(dataset_drift) if dataset_drift is not None else None,
            "share_drifted": float(share) if share is not None else None,
            "drift_share_threshold": self.drift_share_threshold,
            "alert": alert,
            "alert_message": (
                "DATA DRIFT ALERT: production inputs/predictions differ from "
                "training reference. Investigate survey population shift or model decay."
                if alert
                else "No significant drift detected against threshold."
            ),
            "ks_fallback": {
                "engine": ks.get("engine"),
                "dataset_drift": ks.get("dataset_drift"),
                "share_drifted": ks.get("share_drifted"),
                "n_drifted": ks.get("n_drifted"),
                "n_checked": ks.get("n_checked"),
            },
            "evidently": {
                "engine": evid.get("engine"),
                "report_path": evid.get("report_path") or evid.get("latest_report_path"),
                "error": evid.get("error"),
            },
            "mean_shifts": self._mean_shifts(ref, current, feature_cols),
        }

        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info(
            "Drift check done alert=%s share=%s report=%s",
            alert,
            share,
            summary["evidently"].get("report_path"),
        )
        return summary

    @staticmethod
    def _mean_shifts(
        ref: pd.DataFrame, cur: pd.DataFrame, columns: Sequence[str]
    ) -> Dict[str, Dict[str, float]]:
        out = {}
        for c in columns:
            if c not in ref.columns or c not in cur.columns:
                continue
            rm = float(ref[c].mean())
            cm = float(cur[c].mean())
            out[c] = {
                "reference_mean": rm,
                "current_mean": cm,
                "delta": cm - rm,
            }
        return out


def run_drift_check(
    build_reference: bool = False,
    model: Any = None,
    feature_order: Optional[Sequence[str]] = None,
    min_current_rows: int = 30,
) -> Dict[str, Any]:
    """CLI/API helper: optional rebuild reference, then drift check."""
    mon = DriftMonitor(min_current_rows=min_current_rows)
    if build_reference or not REFERENCE_PATH.exists():
        mon.set_reference_from_feast(model=model, feature_order=feature_order)
    return mon.check_drift()
