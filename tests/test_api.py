"""
API tests (Section 7) — FastAPI TestClient, no live uvicorn required.

Uses a stub model + synthetic feature_order so CI does not need multi-MB joblibs.
Set MLFLOW_LOAD_REGISTRY=0 before importing the app.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force local / no-registry load path before importing main
os.environ["MLFLOW_LOAD_REGISTRY"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

import src.api.main as api_main  # noqa: E402


class _StubModel:
    """Minimal sklearn-like estimator for /predict smoke tests."""

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        # Stable non-trivial probability
        return np.tile(np.array([[0.35, 0.65]]), (n, 1))

    def predict(self, X: Any) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _tiny_feature_order() -> List[str]:
    return [
        "Workload_A1",
        "Workload_A2",
        "Learners_B1",
        "WLB_D1",
        "Perf_A1",
        "Time_B1",
        "Emot_C3",
        "Fin_A1",
        "Fin_D4",
        "Soc_A1",
        "Soc_D1",
        "Tech_Awareness",
        "Tech_Platforms_Not_applicable",
        "Age_Category_25_34_",
        "Age_Category_35_44",
        "Gender_Female",
        "Gender_Male_",
        "School_Type_Primary_School",
        "School_Type_Secondary_School",
        "Years_in_Service_5_10_years",
        "Education_Qualification_Bachelor_s_Degree",
        "ICT_Skills_Basic",
    ]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    """App client with stub model injected (CI-safe)."""
    fo = _tiny_feature_order()
    monkeypatch.setattr(api_main, "MODEL", _StubModel())
    monkeypatch.setattr(api_main, "FEATURE_ORDER", fo)
    monkeypatch.setattr(api_main, "MODEL_NAME", "stub-model")
    monkeypatch.setattr(api_main, "MODEL_SOURCE", "test_stub")
    monkeypatch.setattr(api_main, "LOAD_ERROR", None)
    monkeypatch.setattr(api_main, "MODEL_REGISTRY_INFO", {"stub": True})

    # Avoid writing monitoring logs into the real data/ folder during unit tests
    class _NoopLogger:
        def log_prediction(self, **kwargs):  # noqa: ANN003
            return None

    monkeypatch.setattr(api_main, "get_prediction_logger", lambda: _NoopLogger())

    with TestClient(api_main.app) as c:
        yield c


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["status"] == "healthy"
    assert body["feature_count"] == len(_tiny_feature_order())


def test_root(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200


def test_model_info(client: TestClient):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["model_source"] == "test_stub"
    assert "ethical_note" in body


def test_features_schema(client: TestClient):
    r = client.get("/features/schema")
    assert r.status_code == 200
    body = r.json()
    assert "workload" in body["section_scores_1_to_5"]
    assert body["full_feature_count"] == len(_tiny_feature_order())


def test_predict_valid_questionnaire(client: TestClient):
    payload = {
        "age_category": "25-34",
        "gender": "Female",
        "years_in_service": "5-10 years",
        "school_type": "Primary School",
        "education_qualification": "Bachelor's Degree",
        "ict_skills": "Basic",
        "workload": 4.0,
        "learners": 3.5,
        "work_life_balance": 4.0,
        "performance": 3.0,
        "time_pressure": 4.0,
        "emotional_impact": 3.5,
        "financial": 4.0,
        "social_support": 3.0,
        "tech_awareness": True,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "high_risk_probability" in body
    assert body["predicted_risk"] in ("High", "Low")
    assert body["validation_passed"] is True
    assert 0.0 <= body["high_risk_probability"] <= 1.0


def test_predict_out_of_range_likert_rejected_by_schema(client: TestClient):
    """Pydantic Field(ge=1, le=5) rejects before domain gatekeeper → HTTP 422."""
    r = client.post(
        "/predict",
        json={"workload": 99.0, "gender": "Female"},
    )
    assert r.status_code == 422


def test_predict_invalid_teacher_id_returns_400(client: TestClient):
    """Domain gatekeeper: teacher_id must be >= 1 → HTTP 400."""
    r = client.post("/predict", json={"teacher_id": -1})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["message"] == "Validation failed"
    assert any("teacher_id" in e.lower() for e in detail["errors"])


def test_validate_questionnaire_endpoint(client: TestClient):
    r = client.post(
        "/validate/questionnaire",
        json={"workload": 3.0, "gender": "Male"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("valid") is True or "valid" in body


def test_predict_503_when_model_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(api_main, "MODEL", None)
    monkeypatch.setattr(api_main, "FEATURE_ORDER", _tiny_feature_order())
    monkeypatch.setattr(api_main, "LOAD_ERROR", "forced missing model")
    with TestClient(api_main.app) as c:
        r = c.post("/predict", json={"workload": 3.0, "gender": "Female"})
        assert r.status_code == 503


@pytest.mark.integration
def test_predict_with_local_artifacts_if_present():
    """
    Optional: exercise real local joblib when artifacts exist.
    Skip in default CI (mark=integration); run with: pytest -m integration
    """
    fo_path = ROOT / "artifacts" / "feature_order.joblib"
    model_path = ROOT / "artifacts" / "meta4.joblib"
    if not fo_path.exists() or not model_path.exists():
        pytest.skip("local artifacts not present")

    import joblib

    os.environ["MLFLOW_LOAD_REGISTRY"] = "0"
    # Re-bind globals to real artifacts for this one test
    api_main.FEATURE_ORDER = list(joblib.load(fo_path))
    api_main.MODEL = joblib.load(model_path)
    api_main.MODEL_NAME = "meta4.joblib"
    api_main.MODEL_SOURCE = "local_artifacts"
    api_main.LOAD_ERROR = None

    class _NoopLogger:
        def log_prediction(self, **kwargs):  # noqa: ANN003
            return None

    api_main.get_prediction_logger = lambda: _NoopLogger()  # type: ignore[assignment]

    with TestClient(api_main.app) as c:
        r = c.post(
            "/predict",
            json={
                "gender": "Female",
                "age_category": "25-34",
                "workload": 4.0,
                "financial": 4.0,
            },
        )
        assert r.status_code == 200, r.text
        assert "high_risk_probability" in r.json()
