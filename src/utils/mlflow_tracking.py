"""
MLflow Tracking + Model Registry helpers
========================================

Components:
- Experiment tracking (params, metrics, artifacts, tags)
- Model Registry (registers new versions; never auto-sets @champion)
- MLflow UI always shows a built-in @latest (= highest version) — that is MLflow, not our code
- Promotion alias @champion is set manually in the MLflow UI only
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

# Defaults aligned with local MLflow server
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
DEFAULT_EXPERIMENT = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "teacher-mental-health-risk"
)
# Base registry family name — append variant via MLFLOW_MODEL_VARIANT (meta1|meta2|meta3|meta4|rf)
REGISTRY_BASE_NAME = "teacher-mental-health-risk"
VALID_MODEL_VARIANTS = ("meta1", "meta2", "meta3", "meta4", "rf")
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
LATEST_ALIAS = "latest"

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_registered_model_name(
    base: Optional[str] = None,
    variant: Optional[str] = None,
) -> str:
    """
    Build a full MLflow registered model name from base + optional variant.

    Env:
      MLFLOW_REGISTERED_MODEL_NAME=teacher-mental-health-risk   # base / family
      MLFLOW_MODEL_VARIANT=meta1                                # optional pin

    If variant is empty, returns the base name only (serving should then
    discover which *-metaN / *-rf currently holds @champion).
    """
    raw_base = (
        base
        if base is not None
        else os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", REGISTRY_BASE_NAME)
    ).strip()
    if variant is not None:
        raw_variant = str(variant).strip().lower()
    else:
        raw_variant = os.environ.get("MLFLOW_MODEL_VARIANT", "").strip().lower()

    known_suffixes = tuple(f"-{v}" for v in VALID_MODEL_VARIANTS)
    if any(raw_base.endswith(s) for s in known_suffixes):
        return raw_base

    if not raw_variant:
        return raw_base or REGISTRY_BASE_NAME

    if raw_variant not in VALID_MODEL_VARIANTS:
        raise ValueError(
            f"Invalid MLFLOW_MODEL_VARIANT={raw_variant!r}. "
            f"Expected one of {VALID_MODEL_VARIANTS}"
        )
    family = raw_base or REGISTRY_BASE_NAME
    return f"{family}-{raw_variant}"


def discover_champion_model(
    *,
    base: Optional[str] = None,
    alias: str = CHAMPION_ALIAS,
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Find the single registered model under ``base`` that has ``alias`` (champion).

    Humans set @champion in the MLflow UI on exactly one of:
      teacher-mental-health-risk-meta1 … meta4 / -rf

    Returns dict: name, version, source, alias, candidates_checked.
    Raises ValueError if none or multiple champions exist.
    """
    _, _, MlflowClient = _import_mlflow()
    configure_mlflow(tracking_uri=tracking_uri, allow_file_fallback=True)
    client = MlflowClient()
    family = (base or os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", REGISTRY_BASE_NAME)).strip()
    alias = (alias or CHAMPION_ALIAS).strip()

    checked: List[str] = []
    found: List[Dict[str, Any]] = []

    # Prefer known variant names first, then any other registry names with the prefix
    candidate_names = [f"{family}-{v}" for v in VALID_MODEL_VARIANTS]
    try:
        for rm in client.search_registered_models(max_results=200):
            if rm.name.startswith(family) and rm.name not in candidate_names:
                candidate_names.append(rm.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list registered models broadly: %s", exc)

    for name in candidate_names:
        checked.append(name)
        try:
            mv = client.get_model_version_by_alias(name, alias)
        except Exception:  # noqa: BLE001
            continue
        found.append(
            {
                "name": name,
                "version": str(mv.version),
                "source": mv.source,
                "alias": alias,
                "run_id": getattr(mv, "run_id", None),
            }
        )

    if not found:
        raise ValueError(
            f"No registered model under '{family}-*' has alias @{alias}. "
            f"In MLflow UI, open one model (meta1…meta4/rf) → Add Alias → {alias}. "
            f"Checked: {checked}"
        )
    if len(found) > 1:
        names = [f["name"] for f in found]
        raise ValueError(
            f"Multiple models have @{alias}: {names}. "
            "Delete the alias from all but the one model you want to serve."
        )
    result = found[0]
    result["candidates_checked"] = checked
    logger.info(
        "Discovered serving model %s@%s (version %s)",
        result["name"],
        alias,
        result["version"],
    )
    return result


DEFAULT_REGISTERED_MODEL = resolve_registered_model_name(
    variant=os.environ.get("MLFLOW_MODEL_VARIANT", "") or None
)


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
            # Normalize Windows backslashes so Linux/Docker can resolve artifact paths
            model_path = str(context.artifacts["model"]).replace("\\", "/")
            self.model = joblib.load(model_path)
            fo_path = context.artifacts.get("feature_order")
            if fo_path:
                fo_path = str(fo_path).replace("\\", "/")
                self.feature_order = list(joblib.load(fo_path))
            else:
                self.feature_order = getattr(self.model, "feature_order", None)

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

    Registers a new model version only. Does not set @champion unless
    set_champion=True (default False). MLflow itself may display a built-in
    @latest alias in the UI (= highest version); that is not production.

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
    alias: str = CHAMPION_ALIAS,
    tracking_uri: Optional[str] = None,
) -> str:
    """
    Build a models:/ URI for a UI-assigned alias (default: champion).

    Passing alias='latest' resolves to the highest version number (MLflow's
    built-in notion of latest). Serving code must not use that for production.
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


def _load_joblib_from_model_version_source(source: str) -> Any:
    """
    Direct joblib load from a model version source directory.
    Survives Windows→Linux path quirks in MLflow pyfunc MLmodel files.
    """
    raw = source.replace("file:///", "").replace("file://", "")
    # Docker shared store rewrite
    host_marker = "Teachers Mental Health/mlruns"
    if host_marker.replace("\\", "/") in raw.replace("\\", "/"):
        # map .../mlruns/... -> /mlflow/mlruns/... when running in compose
        idx = raw.replace("\\", "/").lower().rfind("/mlruns/")
        if idx >= 0:
            candidate = Path("/mlflow") / raw.replace("\\", "/")[idx + 1 :]
            if candidate.exists() or (candidate.parent / "artifacts" / "model.joblib").exists():
                raw = str(candidate)
    root = Path(raw.replace("\\", "/"))
    candidates = [
        root / "artifacts" / "model.joblib",
        root / "model.joblib",
        root,
    ]
    for c in candidates:
        if c.is_file() and c.suffix == ".joblib":
            return joblib.load(c)
        if c.is_dir():
            job = c / "artifacts" / "model.joblib"
            if job.exists():
                return joblib.load(job)
    raise FileNotFoundError(f"No model.joblib under registry source: {source}")


def load_model_from_registry(
    model_name: Optional[str] = None,
    alias: str = CHAMPION_ALIAS,
    tracking_uri: Optional[str] = None,
) -> Any:
    """
    Load a model from the registry by alias (default: champion).

    Tries MLflow pyfunc first; on Windows→Linux path failures, loads the
    underlying joblib from the version source directory under /mlflow/mlruns.
    """
    mlflow, mlflow_pyfunc, MlflowClient = _import_mlflow()
    configure_mlflow(tracking_uri=tracking_uri, allow_file_fallback=True)
    name = model_name or DEFAULT_REGISTERED_MODEL
    uri = resolve_model_uri(model_name=name, alias=alias, tracking_uri=tracking_uri)
    logger.info("Loading model from registry: %s", uri)

    try:
        loaded = mlflow_pyfunc.load_model(uri)
        return RegistryModelAdapter(loaded, source=uri)
    except Exception as exc:  # noqa: BLE001
        logger.warning("pyfunc load failed (%s); trying direct joblib from version source", exc)
        client = MlflowClient()
        if alias is None or str(alias).lower() in ("", "latest"):
            versions = client.search_model_versions(f"name='{name}'")
            if not versions:
                raise
            mv = max(versions, key=lambda v: int(v.version))
        else:
            mv = client.get_model_version_by_alias(name, alias)
        model = _load_joblib_from_model_version_source(mv.source)
        # Attach feature_order from sibling artifact when possible
        raw = str(mv.source).replace("file:///", "").replace("file://", "").replace("\\", "/")
        if "/mlruns/" in raw and not raw.startswith("/mlflow/"):
            idx = raw.find("/mlruns/")
            raw = "/mlflow" + raw[idx:]
        fo = Path(raw) / "artifacts" / "feature_order.joblib"
        if fo.exists() and not getattr(model, "feature_order", None):
            try:
                model.feature_order = list(joblib.load(fo))
            except Exception:  # noqa: BLE001
                pass
        logger.info("Loaded champion via direct joblib from %s", mv.source)
        return model


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
    alias: str = CHAMPION_ALIAS,
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Return registry metadata for health / model-info endpoints (default: champion)."""
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
