"""
Model Training Pipeline
=======================

Trains Meta-Models 1–4 (and optional RF) equally:
  - Local joblib artifacts
  - MLflow Tracking (params + full metrics including accuracy/precision/recall)
  - MLflow Model Registry registration with @latest only

No model is auto-promoted to production. Assign @champion (or any alias)
in the MLflow UI after comparing runs.

Usage:
  python src/models/train.py --data_path survey.xlsx --model meta1
  python src/models/train.py --data_path survey.xlsx --model all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.processing import (  # noqa: E402
    engineer_features_and_target,
    load_and_clean_raw_data,
    prepare_for_modeling,
)
from src.models.meta_models import (  # noqa: E402
    META_DESCRIPTIONS,
    REGISTERED_MODEL_NAMES,
    MetaModel4Config,
    evaluate_binary_model,
    metrics_for_mlflow,
    train_meta_model1,
    train_meta_model2,
    train_meta_model3,
    train_meta_model4,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_engineered(
    data_path: Optional[str] = None,
    use_feast: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], pd.DataFrame]:
    """
    Load modeling matrices.

    use_feast=True → offline features via Feast (or parquet fallback) so train
    and serve share the same feature contract.
    """
    if use_feast:
        from src.data.feast_client import get_training_matrix
        import pandas as pd

        X, y, feature_order, entity_df = get_training_matrix(use_feast_historical=True)
        # PHQ for meta4 sample weights
        feast_parquet = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "feast"
            / "teacher_survey_features.parquet"
        )
        if feast_parquet.exists():
            raw = pd.read_parquet(feast_parquet)
            # align by teacher_id order from entity_df
            if "teacher_id" in entity_df.columns and "teacher_id" in raw.columns:
                merged = entity_df[["teacher_id"]].merge(
                    raw[["teacher_id", "phq_total"]],
                    on="teacher_id",
                    how="left",
                )
                phq = merged["phq_total"].astype(float)
                phq.index = X.index
            else:
                phq = raw["phq_total"].astype(float).reset_index(drop=True)
                phq.index = X.index
        else:
            phq = pd.Series(np.full(len(X), 10.0), index=X.index, name="PHQ_Total")
        logger.info(
            "Loaded features via Feast offline path: X=%s high_risk=%.1f%%",
            X.shape,
            100 * float(y.mean()),
        )
        return X, y, phq, feature_order, entity_df

    if not data_path:
        raise ValueError("data_path is required when use_feast=False")

    df_raw = load_and_clean_raw_data(data_path)
    df = engineer_features_and_target(df_raw)
    X, y, feature_order = prepare_for_modeling(df)
    if y is None:
        raise ValueError("Target High_Risk missing after feature engineering.")
    if "PHQ_Total" not in df.columns:
        raise ValueError("PHQ_Total required for Meta-Model 4 sample weighting.")
    phq = df["PHQ_Total"].copy().loc[X.index]
    return X, y, phq, feature_order, df

def _log_metrics_block(title: str, metrics: Dict[str, Any]) -> None:
    logger.info(
        "\n%s\n  accuracy=%.4f  precision(HR)=%.4f  recall(HR)=%.4f\n"
        "  precision_macro=%.4f  recall_macro=%.4f\n"
        "  f1_high_risk=%.4f  macro_f1=%.4f\n"
        "  roc_auc=%.4f  AP=%.4f  threshold=%.2f",
        title,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_high_risk"],
        metrics["macro_f1"],
        metrics["roc_auc"],
        metrics["average_precision"],
        metrics.get("threshold", 0.5),
    )


def _mlflow_log(
    *,
    model: Any,
    feature_order: List[str],
    model_key: str,
    params: Dict[str, Any],
    test_metrics: Dict[str, Any],
    train_metrics: Dict[str, Any],
    artifact_paths: Dict[str, Path],
    X_example: pd.DataFrame,
    use_mlflow: bool,
    mlflow_tracking_uri: Optional[str],
    mlflow_experiment: Optional[str],
    registered_model_name: Optional[str],
) -> Dict[str, Any]:
    """Register model equally: @latest only. No champion/challenger in code."""
    if not use_mlflow:
        return {}
    from src.utils.mlflow_tracking import log_training_run

    reg = registered_model_name or REGISTERED_MODEL_NAMES.get(
        model_key, f"teacher-mental-health-risk-{model_key}"
    )
    metrics = {
        **metrics_for_mlflow("test", test_metrics),
        **metrics_for_mlflow("train", train_metrics),
    }
    return log_training_run(
        model=model,
        feature_order=feature_order,
        params=params,
        metrics=metrics,
        tags={
            "model_family": model_key,
            "target": "High_Risk",
            "description": META_DESCRIPTIONS.get(model_key, model_key),
            "promotion": "manual_via_mlflow_ui_only",
        },
        artifact_paths=artifact_paths,
        registered_model_name=reg,
        run_name=f"{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        input_example=X_example,
        set_champion=False,
        set_challenger=False,
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
    )


def train_meta_pipeline(
    model_key: str,
    data_path: Optional[str] = None,
    output_dir: str = "artifacts",
    random_state: int = 42,
    test_size: float = 0.2,
    use_mlflow: bool = True,
    use_feast: bool = False,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    registered_model_name: Optional[str] = None,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Train one of meta1|meta2|meta3|meta4 equally.

    use_feast=True reads the offline feature matrix from Feast/parquet
    (run feast_pipeline first) so features match online serving.
    """
    if model_key not in ("meta1", "meta2", "meta3", "meta4"):
        raise ValueError(f"Unknown meta model: {model_key}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TRAINING %s — %s", model_key.upper(), META_DESCRIPTIONS[model_key])
    logger.info("Feature source: %s", "Feast offline" if use_feast else "processing.py")
    logger.info("=" * 60)

    X, y, phq, feature_order, _df = _load_engineered(
        data_path=data_path, use_feast=use_feast
    )
    joblib.dump(feature_order, output_path / "feature_order.joblib")
    logger.info("Features: %s | High_Risk rate: %.1f%%", X.shape, 100 * y.mean())

    # Optional batch quality gate before training (GE or pandas fallback)
    try:
        from src.data.data_validation import validate_batch_features

        batch_df = X.copy()
        batch_df["high_risk"] = y.values
        if phq is not None:
            batch_df["phq_total"] = phq.values
        batch = validate_batch_features(
            batch_df, feature_order=feature_order, include_label=True
        )
        logger.info(
            "Training-data validation: %s/%s passed (rate=%.1f%%, engine=%s)",
            batch.get("passed"),
            batch.get("total"),
            100 * float(batch.get("pass_rate") or 0),
            batch.get("engine"),
        )
        if not batch.get("success"):
            logger.warning(
                "Training data failed some quality checks — review before trusting metrics"
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Training-data validation skipped: %s", exc)
    X_train, X_test, y_train, y_test, phq_train, _phq_test = train_test_split(
        X, y, phq, test_size=test_size, random_state=random_state, stratify=y
    )

    train_info: Dict[str, Any] = {}
    if model_key == "meta1":
        result = train_meta_model1(
            X_train, y_train, random_state=random_state, logger_=logger
        )
        model = result.model
        train_info = result.train_info or {}
    elif model_key == "meta2":
        result = train_meta_model2(
            X_train, y_train, random_state=random_state, logger_=logger
        )
        model = result.model
        train_info = result.train_info or {}
    elif model_key == "meta3":
        result = train_meta_model3(
            X_train,
            y_train,
            X_test=X_test,
            y_test=y_test,
            random_state=random_state,
            logger_=logger,
        )
        model = result.model
        train_info = result.train_info or {}
    else:
        cfg = MetaModel4Config(random_state=random_state, test_size=test_size)
        model, train_info = train_meta_model4(
            X_train=X_train,
            y_train=y_train,
            phq_train=phq_train,
            feature_order=feature_order,
            cfg=cfg,
            logger=logger,
        )

    test_metrics = evaluate_binary_model(model, X_test, y_test)
    train_metrics = evaluate_binary_model(model, X_train, y_train)

    logger.info(
        "\n%s",
        classification_report(
            y_test, model.predict(X_test), target_names=["Low Risk", "High Risk"]
        ),
    )
    _log_metrics_block(f"TEST — {model_key}", test_metrics)
    _log_metrics_block(f"TRAIN — {model_key}", train_metrics)

    artifact_name = f"{model_key}.joblib"
    # Keep meta_model4.joblib alias for older paths
    joblib.dump(model, output_path / artifact_name)
    if model_key == "meta4":
        joblib.dump(model, output_path / "meta_model4.joblib")
        if hasattr(model, "final_xgb"):
            joblib.dump(model.final_xgb, output_path / "final_xgb.joblib")
            joblib.dump(model.cal_rf, output_path / "cal_rf.joblib")
            joblib.dump(model.cal_xgb, output_path / "cal_xgb.joblib")
            joblib.dump(model.cal_cat, output_path / "cal_cat.joblib")

    reg_name = registered_model_name or REGISTERED_MODEL_NAMES[model_key]
    metadata: Dict[str, Any] = {
        "model_type": model_key,
        "description": META_DESCRIPTIONS[model_key],
        "n_features": len(feature_order),
        "registered_model_name": reg_name,
        "metrics_test": {
            k: round(float(v), 4)
            for k, v in test_metrics.items()
            if isinstance(v, (int, float))
        },
        "metrics_train": {
            k: round(float(v), 4)
            for k, v in train_metrics.items()
            if isinstance(v, (int, float))
        },
        "train_info": train_info,
        "artifact": artifact_name,
        "target_definition": "High_Risk = PHQ_Total >= 15",
        "random_state": random_state,
        "test_size": test_size,
        "note": "No auto production alias. Promote in MLflow UI only.",
    }

    mlflow_info: Dict[str, Any] = {}
    try:
        params = {
            "model_type": model_key,
            "description": META_DESCRIPTIONS[model_key],
            "random_state": random_state,
            "test_size": test_size,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "n_features": len(feature_order),
            "high_risk_prevalence_train": float(y_train.mean()),
            "data_path": str(data_path) if data_path else "feast_offline",
            "feature_source": "feast" if use_feast else "processing",
        }
        for k, v in (train_info or {}).items():
            if isinstance(v, (int, float, str, bool)):
                params[f"info_{k}"] = v

        mlflow_info = _mlflow_log(
            model=model,
            feature_order=feature_order,
            model_key=model_key,
            params=params,
            test_metrics=test_metrics,
            train_metrics=train_metrics,
            artifact_paths={
                model_key: output_path / artifact_name,
                "feature_order": output_path / "feature_order.joblib",
            },
            X_example=X_train.head(3),
            use_mlflow=use_mlflow,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
            registered_model_name=reg_name,
        )
        if mlflow_info:
            metadata["mlflow"] = mlflow_info
    except Exception as exc:  # noqa: BLE001
        logger.exception("MLflow logging failed for %s: %s", model_key, exc)
        metadata["mlflow_error"] = str(exc)

    with open(output_path / f"model_metadata_{model_key}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    joblib.dump(metadata, output_path / f"model_metadata_{model_key}.joblib")
    if model_key == "meta4":
        joblib.dump(metadata, output_path / "model_metadata.joblib")
        with open(output_path / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    logger.info(
        "Done %s → %s | registry=%s | mlflow_run=%s",
        model_key,
        artifact_name,
        reg_name,
        mlflow_info.get("run_id"),
    )
    return model, feature_order, metadata


def train_all_meta_models(
    data_path: Optional[str] = None,
    output_dir: str = "artifacts",
    random_state: int = 42,
    test_size: float = 0.2,
    use_mlflow: bool = True,
    use_feast: bool = False,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    registered_model_name: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Train meta1–4 equally; write comparison JSON; optional MLflow comparison run."""
    results: Dict[str, Dict[str, Any]] = {}
    common = dict(
        data_path=data_path,
        output_dir=output_dir,
        random_state=random_state,
        test_size=test_size,
        use_mlflow=use_mlflow,
        use_feast=use_feast,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        registered_model_name=None,  # each model keeps its own registry name
    )
    for key in ("meta1", "meta2", "meta3", "meta4"):
        _m, _fo, meta = train_meta_pipeline(model_key=key, **common)
        results[key] = meta

    comparison = {k: v.get("metrics_test", {}) for k, v in results.items()}
    out = Path(output_dir)
    with open(out / "meta_models_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparison:\n%s", json.dumps(comparison, indent=2))

    if use_mlflow:
        try:
            from src.utils.mlflow_tracking import _import_mlflow, configure_mlflow

            configure_mlflow(
                tracking_uri=mlflow_tracking_uri, experiment_name=mlflow_experiment
            )
            mlflow, _, _ = _import_mlflow()
            with mlflow.start_run(run_name="meta_models_comparison"):
                for k, m in comparison.items():
                    for metric_name, val in m.items():
                        if isinstance(val, (int, float)):
                            mlflow.log_metric(f"{k}_{metric_name}", float(val))
                mlflow.log_artifact(str(out / "meta_models_comparison.json"))
                mlflow.set_tags(
                    {
                        "run_type": "comparison",
                        "models": "meta1,meta2,meta3,meta4",
                        "promotion": "manual_via_mlflow_ui_only",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Comparison MLflow run failed: %s", exc)
    return results


def train_rf_baseline(
    data_path: Optional[str] = None,
    output_dir: str = "artifacts",
    random_state: int = 42,
    use_mlflow: bool = True,
    use_feast: bool = False,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment: Optional[str] = None,
    registered_model_name: Optional[str] = None,
) -> Tuple[Any, List[str]]:
    """Optional RF baseline — same equal MLflow treatment."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X, y, _phq, feature_order, _df = _load_engineered(
        data_path=data_path, use_feast=use_feast
    )
    joblib.dump(feature_order, output_path / "feature_order.joblib")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    minority = int(y_train.value_counts().get(1, 0))
    majority_target = max(int(minority * 0.85), minority)
    bsmote = BorderlineSMOTE(
        sampling_strategy={0: majority_target} if majority_target > 0 else "auto",
        random_state=random_state,
    )
    try:
        X_res, y_res = bsmote.fit_resample(X_train, y_train)
    except ValueError:
        X_res, y_res = X_train, y_train

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_distributions={
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        },
        n_iter=20,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
        scoring="f1_macro",
        n_jobs=-1,
        random_state=random_state,
        verbose=0,
    )
    search.fit(X_res, y_res)
    best_rf = search.best_estimator_
    test_metrics = evaluate_binary_model(best_rf, X_test, y_test)
    train_metrics = evaluate_binary_model(best_rf, X_train, y_train)
    _log_metrics_block("TEST — rf", test_metrics)

    joblib.dump(best_rf, output_path / "baseline_rf_tuned.joblib")
    joblib.dump(best_rf, output_path / "rf.joblib")

    try:
        _mlflow_log(
            model=best_rf,
            feature_order=feature_order,
            model_key="rf",
            params={
                "model_type": "rf",
                "random_state": random_state,
                **{f"best_{k}": v for k, v in search.best_params_.items()},
                "n_features": len(feature_order),
            },
            test_metrics=test_metrics,
            train_metrics=train_metrics,
            artifact_paths={
                "rf": output_path / "rf.joblib",
                "feature_order": output_path / "feature_order.joblib",
            },
            X_example=X_train.head(3),
            use_mlflow=use_mlflow,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment=mlflow_experiment,
            registered_model_name=registered_model_name
            or REGISTERED_MODEL_NAMES["rf"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("MLflow RF log failed: %s", exc)

    return best_rf, feature_order


# Backward-compatible aliases
def train_meta_model4_pipeline(**kwargs):
    kwargs.setdefault("model_key", "meta4")
    return train_meta_pipeline(**kwargs)


def train_meta_model_n_pipeline(model_key: str, **kwargs):
    return train_meta_pipeline(model_key=model_key, **kwargs)


def train_baseline_model(data_path: str, output_dir: str = "artifacts", random_state: int = 42):
    model, fo, _ = train_meta_pipeline(
        model_key="meta4",
        data_path=data_path,
        output_dir=output_dir,
        random_state=random_state,
    )
    return model, fo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train mental-health meta models (equal MLflow registration)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Raw TSC survey Excel (required unless --use-feast)",
    )
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument(
        "--model",
        type=str,
        choices=["meta1", "meta2", "meta3", "meta4", "rf", "all"],
        default="meta4",
        help="Which model to train (all = meta1–4 equally)",
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--use-feast",
        action="store_true",
        help="Train on Feast offline features (run feast_pipeline first)",
    )
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None)
    parser.add_argument("--mlflow-experiment", type=str, default=None)
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default=None,
        help="Override registry name for a single-model run",
    )
    args = parser.parse_args()

    if not args.use_feast and not args.data_path:
        parser.error("--data_path is required unless --use-feast is set")

    common = dict(
        data_path=args.data_path,
        output_dir=args.output_dir,
        random_state=args.random_state,
        use_mlflow=not args.no_mlflow,
        use_feast=args.use_feast,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        registered_model_name=args.registered_model_name,
    )

    if args.model == "all":
        train_all_meta_models(test_size=args.test_size, **common)
    elif args.model == "rf":
        train_rf_baseline(**common)
    else:
        train_meta_pipeline(model_key=args.model, test_size=args.test_size, **common)