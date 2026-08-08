"""
Production prediction logger
============================

Appends every successful /predict call to a local parquet/CSV log so
Evidently can compare **current (production)** vs **reference (training)** data.

Without logging production inputs/predictions, drift monitoring is invisible.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITOR_DIR = PROJECT_ROOT / "data" / "monitoring"
PRODUCTION_LOG = MONITOR_DIR / "production_predictions.parquet"
PRODUCTION_CSV = MONITOR_DIR / "production_predictions.csv"

# Domain aggregates + score columns used for stable drift monitoring
# (full 222-dim vector is stored as JSON for audit; drift uses compact cols)
DOMAIN_LOG_COLUMNS = [
    "workload_mean",
    "learners_mean",
    "wlb_mean",
    "perf_mean",
    "time_mean",
    "emot_mean",
    "fin_mean",
    "soc_mean",
    "tech_awareness",
    "prediction_proba",
    "prediction",
]


def _means_from_feature_dict(features: Dict[str, float]) -> Dict[str, float]:
    """Compute domain means from a full engineered feature dict."""
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
    out: Dict[str, float] = {}
    for name, prefix in prefixes.items():
        vals = [
            float(v)
            for k, v in features.items()
            if k.startswith(prefix) and k != "Fin_D4"
        ]
        out[name] = float(sum(vals) / len(vals)) if vals else 3.0
    out["tech_awareness"] = float(features.get("Tech_Awareness", 0.0))
    return out


class PredictionLogger:
    """Thread-safe append-only production log for monitoring."""

    def __init__(
        self,
        log_path: Optional[Path] = None,
        max_rows_in_memory: int = 50_000,
    ):
        self.log_path = Path(log_path or PRODUCTION_LOG)
        self.csv_path = self.log_path.with_suffix(".csv")
        self.max_rows = max_rows_in_memory
        self._lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        *,
        features: Dict[str, float],
        prediction: int,
        prediction_proba: float,
        model_name: str = "unknown",
        source: str = "api",
        teacher_id: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one production prediction record."""
        domain = _means_from_feature_dict(features)
        row = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "teacher_id": teacher_id,
            "model_name": model_name,
            "source": source,
            "prediction": int(prediction),
            "prediction_proba": float(prediction_proba),
            **domain,
            # Compact audit trail (not all 222 cols as separate parquet fields)
            "feature_snapshot_json": json.dumps(
                {k: float(v) for k, v in list(features.items())[:80]}
            ),
        }
        if extra:
            for k, v in extra.items():
                if k not in row:
                    row[k] = v

        with self._lock:
            df_new = pd.DataFrame([row])
            if self.log_path.exists():
                try:
                    prev = pd.read_parquet(self.log_path)
                    df = pd.concat([prev, df_new], ignore_index=True)
                except Exception:  # noqa: BLE001
                    df = df_new
            else:
                df = df_new

            if len(df) > self.max_rows:
                df = df.iloc[-self.max_rows :].reset_index(drop=True)

            df.to_parquet(self.log_path, index=False)
            # CSV mirror for easy inspection
            try:
                df.drop(columns=["feature_snapshot_json"], errors="ignore").to_csv(
                    self.csv_path, index=False
                )
            except Exception:  # noqa: BLE001
                pass

        logger.debug(
            "Logged prediction proba=%.3f pred=%s n_log=%s",
            prediction_proba,
            prediction,
            "ok",
        )

    def load_production(self, min_rows: int = 1) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(self.log_path)
        if len(df) < min_rows:
            return df
        return df

    def count(self) -> int:
        if not self.log_path.exists():
            return 0
        try:
            return int(len(pd.read_parquet(self.log_path)))
        except Exception:  # noqa: BLE001
            return 0

    def clear(self) -> None:
        with self._lock:
            for p in (self.log_path, self.csv_path):
                if p.exists():
                    p.unlink()


_logger_singleton: Optional[PredictionLogger] = None


def get_prediction_logger() -> PredictionLogger:
    global _logger_singleton
    if _logger_singleton is None:
        _logger_singleton = PredictionLogger()
    return _logger_singleton
