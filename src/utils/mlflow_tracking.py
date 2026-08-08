"""
MLflow Tracking + Model Registry helpers
========================================

Components:
- Experiment tracking (params, metrics, artifacts, tags)
- Model Registry (registers versions; only sets @latest in code)
- Promotion aliases (@champion etc.) are set manually in the MLflow UI
- pyfunc wrapper for sklearn-like / MetaModel4 packages

Default local server (start separately):
    mlflow server --host 0.0.0.0 --port 5000 \\
        --backend-store-uri sqlite:///mlflow.db \\
        --default-artifact-root ./mlruns
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Defaults aligned with local MLflow server from the tutorial
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEFAULT_EXPERIMENT = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "teacher-mental-health-risk"
)
DEFAULT_REGISTERED_MODEL = os.environ.get(
    "MLFLOW_REGISTERED_MODEL_NAME", "teacher-mental-health-risk-meta4"
)
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
LATEST_ALIAS = "latest"

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _import_mlflow():
    try:
        import mlflow
        import mlflow.pyfunc
        from mlflow.tracking import MlflowClient

        return mlflow, mlflow.pyfunc, MlflowClient
    except ImportError as e:
        raise ImportError(
            "mlflow is required. Install with: pip install mlflow"
        ) from e


def configure_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    allow_file_fallback: bool = True,
) -> str:
    """
    Point the client at the tracking server (or local file store).

    Returns the tracking URI actually used.
    """
    mlflow, _, _ = _import_mlflow()
    uri = tracking_uri or DEFAULT_TRACKING_URI
    exp = experiment_name or DEFAULT_EXPERIMENT

    mlflow.set_tracking_uri(uri)

    # Probe server; fall back to local ./mlruns if unreachable
    if uri.startswith("http") and allow_file_fallback:
        try:
            mlflow.search_experiments(max_results=1)
        except Exception as exc:  # noqa: BLE001
            fallback = (PROJECT_ROOT / "mlruns").resolve().as_uri()
            logger.warning(
                "MLflow server unreachable at %s (%s). "
                "Falling back to file store: %s",
                uri,
                exc,
                fallback,
            )
            uri = fallback
            mlflow.set_tracking_uri(uri)

    mlflow.set_experiment(exp)
    logger.info("MLflow tracking URI=%s experiment=%s", uri, exp)
    return uri


def _build_pyfunc_class():
    """Build PythonModel subclass (lazy so import works without mlflow installed)."""
    _, mlflow_pyfunc, _ = _import_mlflow()

    class MetaModel4PyFunc(mlflow_pyfunc.PythonModel):
        """
        MLflow pyfunc wrapper around MetaModel4 (or any predict/predict_proba model).

        Artifacts expected in context:
          - model: joblib path of MetaModel4 / sklearn estimator
          - feature_order: joblib list of feature column names
        """

        def load_context(self, context):
            self.model = joblib.load(context.artifacts["model"])
            fo_path = context.artifacts.get("feature_order")
            self.feature_order: Optional[List[str]] = (
                joblib.load(fo_path)
                if fo_path
                else getattr(self.model, "feature_order", None)
            )

        def _align(self, model_input: Union[pd.DataFrame, np.ndarray, List]) -> Any:
            if isinstance(model_input, pd.DataFrame):
                if self.feature_order is not None:
                    missing = [
                        c for c in self.feature_order if c not in model_input.columns
                    ]
                    for c in missing:
                        model_input[c] = 0.0
                    return model_input[self.feature_order]
                return model_input
            arr = np.asarray(model_input, dtype="float32")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if self.feature_order is not None and arr.shape[1] == len(self.feature_order):
                return pd.DataFrame(arr, columns=self.feature_order)
            return arr

        def predict(self, context, model_input, params=None):  # noqa: ARG002
            X = self._align(model_input)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)
                if (
                    isinstance(proba, np.ndarray)
                    and proba.ndim == 2
                    and proba.shape[1] >= 2
                ):
                    return pd.DataFrame(
                        {
                            "high_risk_probability": proba[:, 1],
                            "predicted_risk": (proba[:, 1] >= 0.5).astype(int),
                        }
                    )
            preds = self.model.predict(X)
            return pd.DataFrame({"predicted_risk": preds})

    return MetaModel4PyFunc


def log_training_run(
    *,
    model: Any,
    feature_order: Sequence[str],
    params: Dict[str, Any],
    metrics: Dict[str, float],
    tags: Optional[Dict[str, str]] = None,
    artifact_paths: Optional[Dict[str, Path]] = None,
    registered_model_name: Optional[str] = None,
    run_name: Optional[str] = None,
    input_example: Optional[pd.DataFrame] = None,
    set_champion: bool = False,
    set_challenger: bool = False,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Log one training run: params, metrics, artifacts, model + registry.

    By default only @latest is set. Champion/challenger promotion is for the
    MLflow UI (or explicit flags) — no model is auto-picked as production.

    Returns dict with run_id, model_uri, version, aliases applied.
    """
    mlflow, mlflow_pyfunc, MlflowClient = _import_mlflow()
    configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)

    reg_name = registered_model_name or DEFAULT_REGISTERED_MODEL
    result: Dict[str, Any] = {
        "run_id": None,
        "model_uri": None,
        "registered_model": reg_name,
        "version": None,
        "aliases": [],
    }

    with mlflow.start_run(run_name=run_name) as run:
        result["run_id"] = run.info.run_id

        # --- params (flatten nested lightly) ---
        flat_params: Dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, (dict, list)):
                flat_params[k] = json.dumps(v, default=str)[:250]
            else:
                flat_params[k] = v
        mlflow.log_params(flat_params)

        # --- metrics ---
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):
                logger.warning("Skipping non-numeric metric %s=%r", k, v)

        # --- tags ---
        base_tags = {
            "project": "teacher-mental-health",
            "pipeline": "week1-2",
            "ethical_note": "screening_tool_not_diagnosis",
        }
        if tags:
            base_tags.update({str(k): str(v) for k, v in tags.items()})
        mlflow.set_tags(base_tags)

        # --- raw joblib artifacts ---
        if artifact_paths:
            for label, path in artifact_paths.items():
                p = Path(path)
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="artifacts")
                    logger.info("Logged artifact %s → %s", label, p.name)

        # --- pyfunc model + registry ---
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model_path = tmp_path / "model.joblib"
            fo_path = tmp_path / "feature_order.joblib"
            joblib.dump(model, model_path)
            joblib.dump(list(feature_order), fo_path)

            # Optional small input example for signature inference
            example = input_example
            if example is None and feature_order:
                example = pd.DataFrame(
                    [{c: 0.0 for c in list(feature_order)[: min(20, len(feature_order))]}]
                )

            PyFuncCls = _build_pyfunc_class()
            logged = mlflow_pyfunc.log_model(
                artifact_path="model",
                python_model=PyFuncCls(),
                artifacts={
                    "model": str(model_path),
                    "feature_order": str(fo_path),
                },
                registered_model_name=reg_name,
                input_example=example,
            )
            result["model_uri"] = logged.model_uri

        client = MlflowClient()
        # Resolve latest version for this registered model
        versions = client.search_model_versions(f"name='{reg_name}'")
        if versions:
            # highest version number
            latest = max(versions, key=lambda v: int(v.version))
            result["version"] = latest.version
            model_version = latest.version

            # Note: MLflow reserves the name "latest" — do not set it manually.
            # Load with models:/name@latest (built-in) or a UI-assigned alias.
            result["aliases"].append("latest (built-in)")

            if set_challenger:
                try:
                    client.set_registered_model_alias(
                        reg_name, CHALLENGER_ALIAS, model_version
                    )
                    result["aliases"].append(CHALLENGER_ALIAS)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not set @challenger alias: %s", exc)

            if set_champion:
                try:
                    client.set_registered_model_alias(
                        reg_name, CHAMPION_ALIAS, model_version
                    )
                    result["aliases"].append(CHAMPION_ALIAS)
                    logger.info(
                        "Promoted %s version %s → @%s",
                        reg_name,
                        model_version,
                        CHAMPION_ALIAS,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not set @champion alias: %s", exc)

        logger.info(
            "MLflow run complete: run_id=%s model=%s v%s aliases=%s",
            result["run_id"],
            reg_name,
            result["version"],
            result["aliases"],
        )
        logger.info(
            "View run: %s/#/experiments — Models: %s",
            mlflow.get_tracking_uri(),
            reg_name,
        )

    return result


def resolve_model_uri(
    model_name: Optional[str] = None,
    alias: str = LATEST_ALIAS,
    tracking_uri: Optional[str] = None,
) -> str:
    """
    Build a models:/ URI.

    'latest' is reserved in MLflow 2.x — we resolve the highest version number
    instead of using @latest. Other aliases (e.g. champion set in the UI) use @alias.
    """
    mlflow, _, MlflowClient = _import_mlflow()
    configure_mlflow(tracking_uri=tracking_uri, allow_file_fallback=True)
    name = model_name or DEFAULT_REGISTERED_MODEL
    client = MlflowClient()

    if alias is None or str(alias).lower() in ("", "latest"):
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            raise ValueError(f"No versions registered for model '{name}'")
        ver = max(versions, key=lambda v: int(v.version)).version
        return f"models:/{name}/{ver}"

    return f"models:/{name}@{alias}"


def load_model_from_registry(
    model_name: Optional[str] = None,
    alias: str = LATEST_ALIAS,
    tracking_uri: Optional[str] = None,
) -> Any:
    """
    Load a model from the registry.

    Prefer an alias assigned in the MLflow UI (e.g. champion).
    Default 'latest' resolves to the highest registered version (no auto-production).
    """
    mlflow, mlflow_pyfunc, _ = _import_mlflow()
    uri = resolve_model_uri(model_name=model_name, alias=alias, tracking_uri=tracking_uri)
    logger.info("Loading model from registry: %s", uri)
    loaded = mlflow_pyfunc.load_model(uri)
    return RegistryModelAdapter(loaded, source=uri)


class RegistryModelAdapter:
    """
    Adapter so FastAPI can call predict / predict_proba on pyfunc models
    the same way as local MetaModel4 joblib packages.
    """

    def __init__(self, pyfunc_model: Any, source: str = "mlflow"):
        self._model = pyfunc_model
        self.source = source
        # unwrap underlying if sklearn flavor
        self._inner = getattr(pyfunc_model, "_model_impl", None)

    def predict(self, X) -> np.ndarray:
        out = self._model.predict(X)
        if isinstance(out, pd.DataFrame):
            if "predicted_risk" in out.columns:
                return out["predicted_risk"].to_numpy()
            return out.iloc[:, -1].to_numpy()
        return np.asarray(out)

    def predict_proba(self, X) -> np.ndarray:
        # Prefer native proba if available on unwrapped model
        for cand in (
            getattr(self._inner, "python_model", None),
            self._inner,
            getattr(self._model, "unwrap_python_model", lambda: None)(),
        ):
            if cand is None:
                continue
            underlying = getattr(cand, "model", cand)
            if hasattr(underlying, "predict_proba"):
                try:
                    return underlying.predict_proba(X)
                except Exception:  # noqa: BLE001
                    pass

        out = self._model.predict(X)
        if isinstance(out, pd.DataFrame) and "high_risk_probability" in out.columns:
            p1 = out["high_risk_probability"].to_numpy(dtype=float)
            return np.column_stack([1.0 - p1, p1])

        # fallback hard labels
        pred = self.predict(X).astype(float)
        return np.column_stack([1.0 - pred, pred])


def get_model_info(
    model_name: Optional[str] = None,
    alias: str = LATEST_ALIAS,
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Return registry metadata for health / model-info endpoints."""
    _, _, MlflowClient = _import_mlflow()
    configure_mlflow(tracking_uri=tracking_uri, allow_file_fallback=True)
    name = model_name or DEFAULT_REGISTERED_MODEL
    client = MlflowClient()
    try:
        if alias is None or str(alias).lower() in ("", "latest"):
            versions = client.search_model_versions(f"name='{name}'")
            if not versions:
                raise ValueError(f"No versions for '{name}'")
            mv = max(versions, key=lambda v: int(v.version))
            resolved = "latest→version"
        else:
            mv = client.get_model_version_by_alias(name, alias)
            resolved = f"alias:{alias}"
        return {
            "registry": "MLflow",
            "model_name": name,
            "alias": alias,
            "resolved_via": resolved,
            "version": mv.version,
            "run_id": mv.run_id,
            "status": mv.status,
            "tracking_uri": tracking_uri or DEFAULT_TRACKING_URI,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "registry": "MLflow",
            "model_name": name,
            "alias": alias,
            "error": str(exc),
            "tracking_uri": tracking_uri or DEFAULT_TRACKING_URI,
        }
