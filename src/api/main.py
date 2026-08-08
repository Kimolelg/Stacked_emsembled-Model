"""
FastAPI — Teacher Mental Health Risk Prediction
================================================

Model loading order:
  1. MLflow registry @champion  (preferred for serving)
  2. MLflow highest registered version (if no champion yet)
  3. Local artifacts (meta4 / meta1–3 / rf)

Prediction UX:
  POST /predict       — compact questionnaire → full vector with defaults
  POST /predict/full  — partial engineered features + defaults

Data validation:
  Runtime gatekeeper before every prediction (HTTP 400 on failure).
  Batch auditor via Great Expectations on Feast/training tables.

Monitoring (Evidently):
  Log production predictions; compare to Feast/training reference for data & prediction drift.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from src.data.data_validation import (
    raise_http_validation_error,
    validate_feature_vector,
    validate_feast_parquet,
    validate_prediction_request,
    validate_questionnaire,
)
from src.monitoring.drift_monitor import DriftMonitor, REPORTS_DIR, run_drift_check
from src.monitoring.prediction_logger import get_prediction_logger

from .feature_builder import (
    expand_questionnaire_to_features,
    features_dict_with_defaults,
    to_model_frame,
)
from .schemas import (
    FullFeaturesInput,
    HealthResponse,
    PredictionOutput,
    SimplePredictionInput,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"

MODEL: Any = None
FEATURE_ORDER: List[str] = []
MODEL_NAME = "none"
MODEL_SOURCE = "none"
MODEL_REGISTRY_INFO: Dict[str, Any] = {}
LOAD_ERROR: Optional[str] = None

MLFLOW_LOAD_REGISTRY = os.environ.get("MLFLOW_LOAD_REGISTRY", "1").lower() in {
    "1",
    "true",
    "yes",
}
MLFLOW_MODEL_NAME = os.environ.get(
    "MLFLOW_REGISTERED_MODEL_NAME", "teacher-mental-health-risk-meta4"
)
# Prefer champion (set in MLflow UI after comparing runs)
MLFLOW_MODEL_ALIAS = os.environ.get("MLFLOW_MODEL_ALIAS", "champion")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def _load_feature_order_local() -> List[str]:
    fo_path = ARTIFACTS_DIR / "feature_order.joblib"
    if fo_path.exists():
        return list(joblib.load(fo_path))
    return []


def _attach_feature_order_from_model() -> None:
    global FEATURE_ORDER
    FEATURE_ORDER = _load_feature_order_local()
    if FEATURE_ORDER:
        return
    try:
        um = MODEL._model.unwrap_python_model()  # type: ignore[attr-defined]
        fo = getattr(um, "feature_order", None)
        if fo:
            FEATURE_ORDER = list(fo)
            return
    except Exception:  # noqa: BLE001
        pass
    fo = getattr(MODEL, "feature_order", None)
    if fo:
        FEATURE_ORDER = list(fo)


def _try_mlflow_alias(alias: str) -> bool:
    """Load models:/{name}@{alias} (or highest version if alias=latest)."""
    global MODEL, MODEL_NAME, MODEL_SOURCE, MODEL_REGISTRY_INFO
    from src.utils.mlflow_tracking import get_model_info, load_model_from_registry

    MODEL = load_model_from_registry(
        model_name=MLFLOW_MODEL_NAME,
        alias=alias,
        tracking_uri=MLFLOW_TRACKING_URI,
    )
    MODEL_REGISTRY_INFO = get_model_info(
        model_name=MLFLOW_MODEL_NAME,
        alias=alias,
        tracking_uri=MLFLOW_TRACKING_URI,
    )
    MODEL_NAME = f"{MLFLOW_MODEL_NAME}@{alias}"
    MODEL_SOURCE = "mlflow_registry"
    _attach_feature_order_from_model()
    logger.info(
        "Loaded MLflow model %s (version=%s, features=%s)",
        MODEL_NAME,
        MODEL_REGISTRY_INFO.get("version"),
        len(FEATURE_ORDER),
    )
    return True


def _load_from_mlflow() -> bool:
    """Prefer @champion, then fall back to highest version."""
    if not MLFLOW_LOAD_REGISTRY:
        return False

    # User-configured alias first (default: champion)
    preferred = MLFLOW_MODEL_ALIAS or "champion"
    attempts = [preferred]
    if preferred.lower() != "latest":
        attempts.append("latest")  # highest version fallback

    last_err: Optional[Exception] = None
    for alias in attempts:
        try:
            return _try_mlflow_alias(alias)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning(
                "MLflow load failed for %s@%s: %s",
                MLFLOW_MODEL_NAME,
                alias,
                exc,
            )
    if last_err:
        logger.warning("All MLflow registry attempts failed: %s", last_err)
    return False


def _load_from_local_artifacts() -> bool:
    global MODEL, FEATURE_ORDER, MODEL_NAME, MODEL_SOURCE, MODEL_REGISTRY_INFO
    FEATURE_ORDER = _load_feature_order_local()

    candidates = [
        "meta4.joblib",
        "meta_model4.joblib",
        "meta3.joblib",
        "meta2.joblib",
        "meta1.joblib",
        "rf.joblib",
        "baseline_rf_tuned.joblib",
    ]
    pinned = os.environ.get("LOCAL_MODEL_FILE")
    if pinned:
        candidates = [pinned] + [c for c in candidates if c != pinned]

    errors: List[str] = []
    for mf in candidates:
        path = ARTIFACTS_DIR / mf
        if not path.exists():
            continue
        try:
            MODEL = joblib.load(path)
            MODEL_NAME = mf
            MODEL_SOURCE = "local_artifacts"
            MODEL_REGISTRY_INFO = {"path": str(path)}
            _attach_feature_order_from_model()
            logger.info("Loaded local model: %s (features=%s)", mf, len(FEATURE_ORDER))
            return True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{mf}: {exc}")
            logger.warning("Could not load %s: %s", mf, exc)

    # Reassemble meta4 components if present
    parts = [
        ARTIFACTS_DIR / "cal_rf.joblib",
        ARTIFACTS_DIR / "cal_xgb.joblib",
        ARTIFACTS_DIR / "cal_cat.joblib",
        ARTIFACTS_DIR / "final_xgb.joblib",
    ]
    if all(p.exists() for p in parts):
        try:
            from src.models.meta_models import MetaModel4

            MODEL = MetaModel4(
                cal_rf=joblib.load(parts[0]),
                cal_xgb=joblib.load(parts[1]),
                cal_cat=joblib.load(parts[2]),
                final_xgb=joblib.load(parts[3]),
                feature_order=FEATURE_ORDER or None,
            )
            MODEL_NAME = "meta4 (reassembled)"
            MODEL_SOURCE = "local_artifacts"
            MODEL_REGISTRY_INFO = {"reassembled": True}
            _attach_feature_order_from_model()
            logger.info("Reassembled Meta-Model 4 from components")
            return True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"reassemble: {exc}")
            logger.warning("Reassemble failed: %s", exc)

    if errors:
        logger.error("Local artifact load failures: %s", "; ".join(errors[:5]))
    return False


try:
    if not _load_from_mlflow() and not _load_from_local_artifacts():
        raise FileNotFoundError(
            "No model found. Start MLflow, train models, set @champion in UI, "
            "or ensure artifacts/*.joblib exist. "
            "See pipeline commands in README."
        )
except Exception as e:
    logger.error("Failed to load model: %s", e)
    FEATURE_ORDER = []
    MODEL = None
    MODEL_NAME = "none"
    MODEL_SOURCE = "none"
    MODEL_REGISTRY_INFO = {"error": str(e)}
    LOAD_ERROR = str(e)

app = FastAPI(
    title="Teacher Mental Health Risk Prediction API",
    description="""
**Screening support tool** (not a clinical diagnosis).

### Easy input
`POST /predict` — demographics + ~8 section scores (1–5), or `teacher_id` (Feast).  
Missing fields use neutral defaults (Likert=3, binary/OHE=0).

### Advanced
`POST /predict/full` — any subset of engineered features; rest defaulted.

### Validation (Great Expectations / domain rules)
Invalid inputs return **HTTP 400** with clear errors before prediction  
(section scores 1–5, binary 0/1, no NaN, sane demographics).

### Monitoring (Evidently)
Successful predictions are logged; compare production vs training (Feast) for data/prediction drift  
via `/monitoring/*` endpoints.

### Model selection
Configured via env (`MLFLOW_REGISTERED_MODEL_NAME`, `MLFLOW_MODEL_ALIAS`).  
Promote models only in the **MLflow UI**.
    """,
    version="1.4.0-monitored",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model():
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. "
                f"source={MODEL_SOURCE}, name={MODEL_NAME}. "
                f"error={LOAD_ERROR or MODEL_REGISTRY_INFO.get('error')}. "
                "Fix: (1) start MLflow server, (2) train models, "
                "(3) in MLflow UI assign alias 'champion' on a version, "
                "(4) restart uvicorn. Or ensure artifacts/*.joblib exist."
            ),
        )
    if not FEATURE_ORDER:
        raise HTTPException(
            status_code=503,
            detail=(
                "feature_order empty. Retrain so artifacts/feature_order.joblib "
                "is written, then restart the API."
            ),
        )
    return MODEL


def _run_prediction(
    model: Any,
    features: Dict[str, float],
    provided: List[str],
    *,
    validation_warnings: Optional[List[str]] = None,
    teacher_id: Optional[int] = None,
) -> PredictionOutput:
    # Final gate: engineered feature vector quality
    fv = validate_feature_vector(features, FEATURE_ORDER)
    if not fv["valid"]:
        raise_http_validation_error(fv, input_snapshot={"provided": provided})

    warnings = list(validation_warnings or []) + list(fv.get("warnings") or [])

    frame = to_model_frame(FEATURE_ORDER, features)
    proba = float(model.predict_proba(frame)[0, 1])
    pred_class = int(model.predict(frame)[0])

    # Production log for Evidently (best-effort; never fail the response)
    try:
        get_prediction_logger().log_prediction(
            features=features,
            prediction=pred_class,
            prediction_proba=proba,
            model_name=MODEL_NAME,
            source=MODEL_SOURCE,
            teacher_id=teacher_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Prediction logging failed (monitoring): %s", exc)

    risk_label = "High" if pred_class == 1 else "Low"
    risk_level = (
        "High Risk (PHQ-9 equivalent ≥15)" if pred_class == 1 else "Low Risk"
    )
    if proba >= 0.85:
        confidence = "Very High"
    elif proba >= 0.70:
        confidence = "High"
    elif proba >= 0.55:
        confidence = "Moderate"
    else:
        confidence = "Low"

    return PredictionOutput(
        high_risk_probability=round(proba, 4),
        predicted_risk=risk_label,
        risk_level=risk_level,
        confidence=confidence,
        features_used=len(FEATURE_ORDER),
        fields_provided=provided,
        defaults_applied=True,
        validation_passed=True,
        validation_warnings=warnings or None,
        model_version=f"{MODEL_SOURCE}:{MODEL_NAME}",
    )
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        feature_count=len(FEATURE_ORDER),
        version="1.4.0-monitored",
        model_source=MODEL_SOURCE,
        model_name=MODEL_NAME,
    )
@app.get("/model-info", tags=["Health"])
async def model_info():
    return {
        "model_source": MODEL_SOURCE,
        "model_name": MODEL_NAME,
        "model_loaded": MODEL is not None,
        "feature_count": len(FEATURE_ORDER),
        "load_error": LOAD_ERROR,
        "registry": MODEL_REGISTRY_INFO,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "preferred_alias": MLFLOW_MODEL_ALIAS,
        "registered_model": MLFLOW_MODEL_NAME,
        "registered_models_equal": [
            "teacher-mental-health-risk-meta1",
            "teacher-mental-health-risk-meta2",
            "teacher-mental-health-risk-meta3",
            "teacher-mental-health-risk-meta4",
            "teacher-mental-health-risk-rf",
        ],
        "load_order": [
            f"MLflow {MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}",
            "MLflow highest version (if champion missing)",
            "local artifacts/*.joblib",
        ],
        "note": "Serving prefers MLflow alias 'champion'. Assign it in the MLflow UI.",
        "ethical_note": "Screening support tool only — not a clinical diagnosis.",
    }

@app.get("/features/schema", tags=["Prediction"])
async def features_schema():
    """Document compact questionnaire fields and default policy."""
    return {
        "endpoint": "POST /predict",
        "section_scores_1_to_5": list(
            [
                "workload",
                "learners",
                "work_life_balance",
                "performance",
                "time_pressure",
                "emotional_impact",
                "financial",
                "social_support",
            ]
        ),
        "demographics": list(
            [
                "age_category",
                "gender",
                "years_in_service",
                "school_type",
                "education_qualification",
                "ict_skills",
            ]
        ),
        "defaults": {
            "likert_items": 3.0,
            "binary_and_one_hot": 0.0,
            "tech_platforms_not_applicable": 1.0,
        },
        "full_feature_count": len(FEATURE_ORDER),
        "full_endpoint": "POST /predict/full",
    }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Prediction"],
    responses={400: {"description": "Validation failed"}},
)
async def predict_simple(
    body: SimplePredictionInput,
    model=Depends(get_model),
):
    """
    Compact questionnaire → risk prediction (validated).

    - With **teacher_id**: features from **Feast** (online store / parquet).
    - Without: questionnaire expansion + defaults.

    Invalid inputs → **HTTP 400** with structured errors (not a silent bad score).
    """
    try:
        payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
        q_val = validate_questionnaire(payload)
        if not q_val["valid"]:
            raise_http_validation_error(q_val, input_snapshot=payload)

        if body.teacher_id is not None:
            try:
                from src.data.feast_client import get_online_feature_row

                features = get_online_feature_row(
                    int(body.teacher_id), feature_order=FEATURE_ORDER
                )
                provided = [f"feast:teacher_id={body.teacher_id}"]
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Feast features unavailable for teacher_id={body.teacher_id}: {exc}. "
                        "Run: python -m src.data.feast_pipeline --data_path <survey.xlsx>"
                    ),
                ) from exc
        else:
            features, provided = expand_questionnaire_to_features(
                FEATURE_ORDER,
                age_category=body.age_category,
                gender=body.gender,
                years_in_service=body.years_in_service,
                school_type=body.school_type,
                education_qualification=body.education_qualification,
                ict_skills=body.ict_skills,
                workload=body.workload,
                learners=body.learners,
                work_life_balance=body.work_life_balance,
                performance=body.performance,
                time_pressure=body.time_pressure,
                emotional_impact=body.emotional_impact,
                financial=body.financial,
                social_support=body.social_support,
                tech_awareness=body.tech_awareness,
                financial_literacy_training=body.financial_literacy_training,
                feature_overrides=body.feature_overrides,
            )
        return _run_prediction(
            model,
            features,
            provided,
            validation_warnings=q_val.get("warnings"),
            teacher_id=body.teacher_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feast/teachers", tags=["Feast"])
async def feast_sample_teachers(limit: int = 20):
    """List sample teacher_ids available in the Feast offline table."""
    try:
        from src.data.feast_client import list_teacher_ids

        ids = list_teacher_ids(limit=limit)
        return {
            "teacher_ids": ids,
            "count": len(ids),
            "usage": "POST /predict with {\"teacher_id\": <id>}",
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

@app.post(
    "/predict/full",
    response_model=PredictionOutput,
    tags=["Prediction"],
    responses={400: {"description": "Validation failed"}},
)
async def predict_full(
    body: FullFeaturesInput,
    model=Depends(get_model),
):
    """
    Partial engineered feature map. Missing keys get defaults.

    Values outside Likert/binary ranges → HTTP 400.
    """
    try:
        raw_features = body.features or {}
        # Pre-check only client-supplied keys before defaults
        if raw_features:
            cleaned = {}
            for k, v in raw_features.items():
                if v is None:
                    continue
                try:
                    cleaned[str(k)] = float(v)
                except (TypeError, ValueError):
                    raise_http_validation_error(
                        {
                            "valid": False,
                            "errors": [f"features[{k!r}] must be numeric (got {v!r})"],
                            "warnings": [],
                        },
                        input_snapshot={"features": raw_features},
                    )
            pre = validate_feature_vector(cleaned, list(cleaned.keys()))
            if not pre["valid"]:
                raise_http_validation_error(
                    pre, input_snapshot={"features": raw_features}
                )
        else:
            pre = {"valid": True, "errors": [], "warnings": []}

        features = features_dict_with_defaults(FEATURE_ORDER, raw_features)
        provided = list(raw_features.keys())
        return _run_prediction(
            model,
            features,
            provided,
            validation_warnings=pre.get("warnings"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate/questionnaire", tags=["Validation"])
async def validate_questionnaire_endpoint(body: SimplePredictionInput):
    """Dry-run validation of a questionnaire payload (no prediction)."""
    payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
    result = validate_questionnaire(payload)
    return {
        "valid": result["valid"],
        "errors": result["errors"],
        "warnings": result.get("warnings") or [],
    }


@app.get("/validate/feast", tags=["Validation"])
async def validate_feast_batch():
    """
    Batch-validate the Feast offline table with Great Expectations (or pandas fallback).

    Use after `feast_pipeline` to audit training-ready data quality.
    """
    result = validate_feast_parquet(feature_order=FEATURE_ORDER or None)
    status = 200 if result.get("success") else 422
    return {
        "status_code_hint": status,
        **result,
    }


# ---------------------------------------------------------------------------
# Monitoring (Evidently)
# ---------------------------------------------------------------------------


@app.get("/monitoring/status", tags=["Monitoring"])
async def monitoring_status():
    """Production log size and whether a reference snapshot exists."""
    from src.monitoring.drift_monitor import REFERENCE_PATH, SUMMARY_PATH

    plog = get_prediction_logger()
    summary = None
    if SUMMARY_PATH.exists():
        import json

        try:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            summary = {"error": "could not parse last summary"}

    return {
        "production_log_rows": plog.count(),
        "production_log_path": str(plog.log_path),
        "reference_exists": REFERENCE_PATH.exists(),
        "reference_path": str(REFERENCE_PATH),
        "last_drift_summary": summary,
        "min_rows_for_drift": 30,
        "hint": (
            "1) POST /monitoring/build-reference  "
            "2) generate traffic via /predict  "
            "3) POST /monitoring/run-drift  "
            "4) GET /monitoring/report"
        ),
    }


@app.post("/monitoring/build-reference", tags=["Monitoring"])
async def monitoring_build_reference(max_rows: int = 5000):
    """
    Build Evidently **reference** snapshot from Feast offline survey features
    (optionally scored with the loaded model for prediction-drift baseline).
    """
    try:
        mon = DriftMonitor()
        info = mon.set_reference_from_feast(
            max_rows=max_rows,
            model=MODEL if MODEL is not None else None,
            feature_order=FEATURE_ORDER or None,
        )
        return {"success": True, **info}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/monitoring/run-drift", tags=["Monitoring"])
async def monitoring_run_drift(
    build_reference: bool = False,
    min_current_rows: int = 30,
):
    """
    Compare production prediction log vs reference (training/Feast).

    Writes HTML report under data/monitoring/reports/ and a JSON summary.
    Returns alert=true if share of drifted columns exceeds threshold.
    """
    try:
        summary = run_drift_check(
            build_reference=build_reference,
            model=MODEL if MODEL is not None else None,
            feature_order=FEATURE_ORDER or None,
            min_current_rows=min_current_rows,
        )
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/monitoring/report", tags=["Monitoring"])
async def monitoring_report():
    """Serve the latest Evidently HTML drift report."""
    latest = REPORTS_DIR / "latest_drift_report.html"
    if not latest.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No drift report yet. Call POST /monitoring/build-reference, "
                "generate /predict traffic, then POST /monitoring/run-drift."
            ),
        )
    return FileResponse(
        path=str(latest),
        media_type="text/html",
        filename="latest_drift_report.html",
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Teacher Mental Health Risk API (validated + monitored)",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "predict": "POST /predict (questionnaire or teacher_id)",
        "predict_full": "POST /predict/full (engineered features + defaults)",
        "validate_questionnaire": "POST /validate/questionnaire",
        "validate_feast": "GET /validate/feast",
        "monitoring_status": "GET /monitoring/status",
        "monitoring_build_reference": "POST /monitoring/build-reference",
        "monitoring_run_drift": "POST /monitoring/run-drift",
        "monitoring_report": "GET /monitoring/report",
        "model_source": MODEL_SOURCE,
        "model_name": MODEL_NAME,
        "validation": "enabled",
        "monitoring": "evidently",
        "warning": "Screening tool only. Not for diagnosis or punitive use.",
    }
