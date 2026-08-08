"""
Meta-Models 1–4 (reproducible training definitions)
====================================================

All stacks from mental_health_final_model.py live here — no preferred production
baseline. Promotion / aliases are managed only in the MLflow UI.

Contents:
  - Meta 1–3: sklearn StackingClassifier variants
  - Meta 4: calibrated RF/XGB/CatBoost + XGB meta with PHQ sample weights
  - evaluate_binary_model: shared metrics for MLflow
  - REGISTERED_MODEL_NAMES: equal registry naming (meta1…meta4, rf)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

ArrayLike = Union[pd.DataFrame, np.ndarray]

# Equal registry names — no model is special-cased as "production"
REGISTERED_MODEL_NAMES = {
    "meta1": "teacher-mental-health-risk-meta1",
    "meta2": "teacher-mental-health-risk-meta2",
    "meta3": "teacher-mental-health-risk-meta3",
    "meta4": "teacher-mental-health-risk-meta4",
    "rf": "teacher-mental-health-risk-rf",
}

META_DESCRIPTIONS = {
    "meta1": "Stack: cal RF + XGB + CatBoost → XGB (raw train; BorderlineSMOTE prepared)",
    "meta2": "Stack: RF + XGB + Cat + LGBM → XGB (BorderlineSMOTE)",
    "meta3": "Stack: cal RF + XGB + Cat + cal LGBM → XGB (SMOTETomek + threshold opt)",
    "meta4": "Custom stack: cal RF/XGB/Cat → XGB meta with PHQ sample weights",
    "rf": "Tuned RandomForest + BorderlineSMOTE",
}


@dataclass
class MetaTrainResult:
    """Trained stack + optional decision threshold + training notes."""

    model: Any
    threshold: float = 0.5
    train_info: Optional[Dict[str, Any]] = None
    display_name: str = "meta"


class ThresholdedModel:
    """Wraps any proba model with a custom decision threshold (Meta-Model 3)."""

    def __init__(self, base_model: Any, threshold: float = 0.5):
        self.base_model = base_model
        self.threshold = float(threshold)

    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def get_params(self, deep: bool = True):
        return {"threshold": self.threshold}


# ---------------------------------------------------------------------------
# Shared evaluation (MLflow metrics)
# ---------------------------------------------------------------------------


def evaluate_binary_model(
    model: Any,
    X: ArrayLike,
    y: ArrayLike,
) -> Dict[str, Any]:
    """
    Binary classification metrics for High_Risk (positive class = 1).

    Logged to MLflow as test_* / train_* with these keys:
      accuracy, precision, recall (positive class),
      precision_macro, recall_macro, f1_high_risk, macro_f1,
      roc_auc, average_precision, threshold
    """
    y_true = np.asarray(y).astype(int)
    y_pred = np.asarray(model.predict(X)).astype(int)
    y_proba = np.asarray(model.predict_proba(X))[:, 1]

    report = classification_report(
        y_true, y_pred, target_names=["Low Risk", "High Risk"], output_dict=True
    )
    return {
        "classification_report": report,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_high_risk": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "threshold": float(getattr(model, "threshold", 0.5)),
    }


def metrics_for_mlflow(prefix: str, metrics: Dict[str, Any]) -> Dict[str, float]:
    """Flatten evaluate_binary_model output into scalar MLflow metric names."""
    keys = (
        "accuracy",
        "precision",
        "recall",
        "precision_macro",
        "recall_macro",
        "f1_high_risk",
        "macro_f1",
        "roc_auc",
        "average_precision",
        "threshold",
    )
    out: Dict[str, float] = {}
    for k in keys:
        if k in metrics and isinstance(metrics[k], (int, float)):
            out[f"{prefix}_{k}"] = float(metrics[k])
    return out


# ---------------------------------------------------------------------------
# Meta-Model 1
# ---------------------------------------------------------------------------


def train_meta_model1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    logger_: Optional[logging.Logger] = None,
) -> MetaTrainResult:
    """BorderlineSMOTE prepared; stack fit on raw train (notebook)."""
    log = logger_.info if logger_ is not None else logger.info
    log("Training Meta-Model 1...")

    bsmote = BorderlineSMOTE(random_state=random_state)
    try:
        X_res, _y_res = bsmote.fit_resample(X_train, y_train)
        resampled_n = int(X_res.shape[0])
    except ValueError as exc:
        log("BorderlineSMOTE failed (%s); using original train", exc)
        resampled_n = int(len(X_train))

    rf_best = RandomForestClassifier(
        n_estimators=400,
        min_samples_split=5,
        min_samples_leaf=2,
        max_depth=10,
        bootstrap=True,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    xgb_best = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )
    cat_best = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        auto_class_weights="Balanced",
        loss_function="Logloss",
        random_seed=random_state,
        verbose=False,
    )
    xgb_best.fit(X_train, y_train)
    cat_best.fit(X_train, y_train)
    cal_rf = CalibratedClassifierCV(rf_best, cv=3, method="isotonic")
    cal_rf.fit(X_train, y_train)

    meta = StackingClassifier(
        estimators=[("rf", cal_rf), ("xgb", xgb_best), ("cat", cat_best)],
        final_estimator=XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        stack_method="predict_proba",
        cv=5,
        n_jobs=1,
        passthrough=False,
    )
    meta.fit(X_train, y_train)
    return MetaTrainResult(
        model=meta,
        train_info={
            "resampling": "BorderlineSMOTE (prepared; stack on raw train)",
            "resampled_n": resampled_n,
            "base_models": ["cal_rf_isotonic", "xgb", "catboost"],
            "meta_learner": "XGBClassifier",
        },
        display_name="meta_model1",
    )


# ---------------------------------------------------------------------------
# Meta-Model 2
# ---------------------------------------------------------------------------


def train_meta_model2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    logger_: Optional[logging.Logger] = None,
) -> MetaTrainResult:
    """BorderlineSMOTE + RF/XGB/Cat/LGBM → XGB stack."""
    log = logger_.info if logger_ is not None else logger.info
    log("Training Meta-Model 2...")

    bsmote = BorderlineSMOTE(random_state=random_state)
    X_res, y_res = bsmote.fit_resample(X_train, y_train)

    estimators = [
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_split=5,
                min_samples_leaf=2,
                max_depth=10,
                bootstrap=True,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        (
            "xgb",
            XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
            ),
        ),
        (
            "cat",
            CatBoostClassifier(
                iterations=1000,
                learning_rate=0.1,
                depth=6,
                auto_class_weights="Balanced",
                loss_function="MultiClass",
                random_seed=random_state,
                verbose=False,
            ),
        ),
        (
            "lgbm",
            LGBMClassifier(
                random_state=random_state, class_weight="balanced", verbose=-1
            ),
        ),
    ]
    meta = StackingClassifier(
        estimators=estimators,
        final_estimator=XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        stack_method="predict_proba",
        cv=5,
        n_jobs=1,
        passthrough=False,
    )
    meta.fit(X_res, y_res)
    return MetaTrainResult(
        model=meta,
        train_info={
            "resampling": "BorderlineSMOTE",
            "resampled_n": int(X_res.shape[0]),
            "base_models": ["rf", "xgb", "catboost", "lgbm"],
            "meta_learner": "XGBClassifier",
        },
        display_name="meta_model2",
    )


# ---------------------------------------------------------------------------
# Meta-Model 3
# ---------------------------------------------------------------------------


def train_meta_model3(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    random_state: int = 42,
    optimize_threshold: bool = True,
    logger_: Optional[logging.Logger] = None,
) -> MetaTrainResult:
    """SMOTETomek + calibrated RF/LGBM + XGB/Cat → XGB; optional thr sweep."""
    log = logger_.info if logger_ is not None else logger.info
    log("Training Meta-Model 3...")

    resampler = SMOTETomek(random_state=random_state)
    X_res, y_res = resampler.fit_resample(X_train, y_train)

    rf_best = RandomForestClassifier(
        n_estimators=400,
        min_samples_split=4,
        min_samples_leaf=1,
        max_depth=12,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    xgb_best = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    cat_best = CatBoostClassifier(
        iterations=800,
        learning_rate=0.07,
        depth=6,
        loss_function="Logloss",
        verbose=False,
        random_seed=random_state,
        auto_class_weights="Balanced",
    )
    lgbm_best = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
    )
    cal_rf = CalibratedClassifierCV(rf_best, cv=3, method="isotonic")
    cal_lgb = CalibratedClassifierCV(lgbm_best, cv=3, method="sigmoid")

    meta = StackingClassifier(
        estimators=[
            ("rf", cal_rf),
            ("xgb", xgb_best),
            ("cat", cat_best),
            ("lgb", cal_lgb),
        ],
        final_estimator=XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),
        stack_method="predict_proba",
        cv=5,
        n_jobs=1,
        passthrough=False,
    )
    meta.fit(X_res, y_res)

    threshold = 0.5
    best_f1 = None
    if optimize_threshold and X_test is not None and y_test is not None:
        y_proba = meta.predict_proba(X_test)[:, 1]
        best_f1 = -1.0
        for t in np.arange(0.40, 0.60, 0.01):
            f1 = f1_score(y_test, (y_proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                threshold = float(t)
        log("Meta-Model 3 threshold=%.2f (test F1=%.4f)", threshold, best_f1)

    return MetaTrainResult(
        model=ThresholdedModel(meta, threshold=threshold),
        threshold=threshold,
        train_info={
            "resampling": "SMOTETomek",
            "resampled_n": int(X_res.shape[0]),
            "base_models": ["cal_rf_isotonic", "xgb", "catboost", "cal_lgbm_sigmoid"],
            "meta_learner": "XGBClassifier",
            "threshold": threshold,
            "threshold_opt_f1": best_f1,
        },
        display_name="meta_model3",
    )


# ---------------------------------------------------------------------------
# Meta-Model 4 (formerly meta_model4.py)
# ---------------------------------------------------------------------------


@dataclass
class MetaModel4Config:
    """Hyperparameters matching the validated notebook FINAL META-MODEL 4."""

    random_state: int = 42
    borderline_phq: float = 15.0
    borderline_weight: float = 0.40
    default_weight: float = 1.00
    test_size: float = 0.2

    rf_n_estimators: int = 300
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    rf_max_depth: int = 10
    rf_bootstrap: bool = True

    xgb_n_estimators: int = 100
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.1

    cat_iterations: int = 1000
    cat_learning_rate: float = 0.1
    cat_depth: int = 6
    cat_loss_function: str = "MultiClass"

    cal_rf_method: str = "isotonic"
    cal_xgb_method: str = "sigmoid"
    cal_cat_method: str = "sigmoid"
    cal_cv: int = 3

    meta_n_estimators: int = 150
    meta_max_depth: int = 3
    meta_learning_rate: float = 0.05
    meta_subsample: float = 0.8
    meta_colsample_bytree: float = 0.8


class MetaModel4:
    """Inference package: calibrated base probs → XGB meta-learner."""

    META_FEATURE_NAMES = ("rf_proba", "xgb_proba", "cat_proba")

    def __init__(
        self,
        cal_rf: CalibratedClassifierCV,
        cal_xgb: CalibratedClassifierCV,
        cal_cat: CalibratedClassifierCV,
        final_xgb: XGBClassifier,
        feature_order: Optional[List[str]] = None,
        threshold: float = 0.5,
        config: Optional[MetaModel4Config] = None,
    ):
        self.cal_rf = cal_rf
        self.cal_xgb = cal_xgb
        self.cal_cat = cal_cat
        self.final_xgb = final_xgb
        self.feature_order = feature_order
        self.threshold = threshold
        self.config = config or MetaModel4Config()

    def _prepare_X(self, X: ArrayLike, dtype: str = "float32") -> ArrayLike:
        if isinstance(X, pd.DataFrame):
            if self.feature_order is not None:
                missing = [c for c in self.feature_order if c not in X.columns]
                if missing:
                    raise ValueError(
                        f"Missing {len(missing)} features (e.g. {missing[:5]})"
                    )
                X = X[self.feature_order]
            return X.astype(dtype)
        arr = np.asarray(X, dtype=dtype)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self.feature_order is not None and arr.shape[1] == len(self.feature_order):
            return pd.DataFrame(arr, columns=self.feature_order)
        return arr

    def meta_features(self, X: ArrayLike) -> np.ndarray:
        X32 = self._prepare_X(X, dtype="float32")
        return np.column_stack(
            [
                self.cal_rf.predict_proba(X32)[:, 1],
                self.cal_xgb.predict_proba(X32)[:, 1],
                self.cal_cat.predict_proba(X32)[:, 1],
            ]
        )

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return self.final_xgb.predict_proba(self.meta_features(X))

    def predict(self, X: ArrayLike) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "feature_order": self.feature_order,
            "n_features": len(self.feature_order) if self.feature_order else None,
            "base_models": list(self.META_FEATURE_NAMES),
            "meta_learner": "XGBClassifier",
        }


def _build_meta4_bases(cfg: MetaModel4Config):
    rf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        min_samples_split=cfg.rf_min_samples_split,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        max_depth=cfg.rf_max_depth,
        bootstrap=cfg.rf_bootstrap,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        random_state=cfg.random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )
    cat = CatBoostClassifier(
        iterations=cfg.cat_iterations,
        learning_rate=cfg.cat_learning_rate,
        depth=cfg.cat_depth,
        auto_class_weights=None,
        loss_function=cfg.cat_loss_function,
        random_seed=cfg.random_state,
        verbose=False,
    )
    return rf, xgb, cat


def resample_with_phq_weights(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    phq_train: pd.Series,
    cfg: MetaModel4Config,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    X_with_phq = pd.concat(
        [
            X_train.reset_index(drop=True),
            phq_train.reset_index(drop=True).rename("PHQ_Total"),
        ],
        axis=1,
    )
    resampler = SMOTETomek(random_state=cfg.random_state)
    X_resampled, y_res = resampler.fit_resample(
        X_with_phq, y_train.reset_index(drop=True)
    )
    if not isinstance(X_resampled, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X_with_phq.columns)
    phq_res = X_resampled["PHQ_Total"]
    X_res = X_resampled.drop(columns=["PHQ_Total"])
    sample_weights = np.where(
        np.isclose(phq_res.to_numpy(dtype=float), cfg.borderline_phq),
        cfg.borderline_weight,
        cfg.default_weight,
    ).astype("float64")
    return X_res, pd.Series(y_res), sample_weights


def train_meta_model4(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    phq_train: pd.Series,
    feature_order: List[str],
    cfg: Optional[MetaModel4Config] = None,
    logger: Optional[Any] = None,
) -> Tuple[MetaModel4, Dict[str, Any]]:
    """Train Meta-Model 4 (PHQ-weighted SMOTETomek + calibrated bases + XGB meta)."""
    cfg = cfg or MetaModel4Config()
    log = logger.info if logger is not None else (lambda *a, **k: None)

    log("SMOTETomek + PHQ sample weights...")
    X_res, y_res, sample_weights = resample_with_phq_weights(
        X_train, y_train, phq_train, cfg
    )
    log(
        "Resampled %s | balance %s",
        X_res.shape,
        pd.Series(y_res).value_counts(normalize=True).to_dict(),
    )

    X_res = X_res[feature_order]
    X_res32 = X_res.astype("float32")
    X_res64 = X_res.astype("float64")
    y_res32 = y_res.astype("float32")
    y_res64 = y_res.astype("float64")
    sw32 = sample_weights.astype("float32")
    sw64 = sample_weights.astype("float64")

    rf, xgb, cat = _build_meta4_bases(cfg)
    log("Fitting calibrated bases...")
    cal_rf = CalibratedClassifierCV(rf, cv=cfg.cal_cv, method=cfg.cal_rf_method)
    cal_xgb = CalibratedClassifierCV(xgb, cv=cfg.cal_cv, method=cfg.cal_xgb_method)
    cal_cat = CalibratedClassifierCV(cat, cv=cfg.cal_cv, method=cfg.cal_cat_method)
    cal_rf.fit(X_res64, y_res64, sample_weight=sw64)
    cal_xgb.fit(X_res32, y_res32, sample_weight=sw32)
    cal_cat.fit(X_res64, y_res64, sample_weight=sw64)

    meta_train = np.column_stack(
        [
            cal_rf.predict_proba(X_res32)[:, 1],
            cal_xgb.predict_proba(X_res32)[:, 1],
            cal_cat.predict_proba(X_res32)[:, 1],
        ]
    )
    final_xgb = XGBClassifier(
        n_estimators=cfg.meta_n_estimators,
        max_depth=cfg.meta_max_depth,
        learning_rate=cfg.meta_learning_rate,
        subsample=cfg.meta_subsample,
        colsample_bytree=cfg.meta_colsample_bytree,
        eval_metric="logloss",
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    final_xgb.fit(meta_train, y_res32, sample_weight=sw32)

    model = MetaModel4(
        cal_rf=cal_rf,
        cal_xgb=cal_xgb,
        cal_cat=cal_cat,
        final_xgb=final_xgb,
        feature_order=list(feature_order),
        threshold=0.5,
        config=cfg,
    )
    info = {
        "resampled_n": int(X_res.shape[0]),
        "n_features": len(feature_order),
        "borderline_weight": cfg.borderline_weight,
        "class_balance_resampled": pd.Series(y_res)
        .value_counts(normalize=True)
        .to_dict(),
        "base_models": ["cal_rf", "cal_xgb", "cal_cat"],
        "meta_learner": "XGBClassifier",
        "resampling": "SMOTETomek + PHQ sample weights",
    }
    return model, info
