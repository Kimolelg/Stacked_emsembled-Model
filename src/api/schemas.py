"""
Pydantic schemas for the Mental Health Risk Prediction API.

Primary path: compact questionnaire (section scores + demographics).
Advanced path: partial engineered features with automatic defaults.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional  # Any used by ValidationErrorResponse

from pydantic import BaseModel, Field

class SimplePredictionInput(BaseModel):
    """
    Compact screening questionnaire (~15 fields).

    Section scores are 1–5 Likert averages broadcast to all items in that domain.
    Omitted fields use safe defaults (Likert=3 Neutral, binary/OHE=0).

    Optional teacher_id: load full engineered vector from Feast online/offline store
    (materialized survey respondent) instead of questionnaire expansion.
    """

    teacher_id: Optional[int] = Field(
        None,
        description=(
            "If set, pull features from Feast (teacher_survey_fv) for this id. "
            "Questionnaire fields are ignored when teacher_id is provided."
        ),
        examples=[1],
    )

    age_category: Optional[str] = Field(
        None,
        description="e.g. '25-34', '35-44', '45-54', '55 and above', 'Below 25'",
        examples=["25-34"],
    )
    gender: Optional[str] = Field(None, examples=["Female"])
    years_in_service: Optional[str] = Field(
        None, examples=["5-10 years"], description="Teaching service band"
    )
    school_type: Optional[str] = Field(
        None, examples=["Primary School", "Secondary School", "Junior School"]
    )
    education_qualification: Optional[str] = Field(
        None, examples=["Bachelor's Degree", "Certificate/Diploma", "Master's Degree"]
    )
    ict_skills: Optional[str] = Field(
        None, examples=["Basic", "Intermediate"]
    )

    workload: Optional[float] = Field(
        None, ge=1, le=5, description="Average workload demand (1–5)"
    )
    learners: Optional[float] = Field(
        None, ge=1, le=5, description="Class size / learners load (1–5)"
    )
    work_life_balance: Optional[float] = Field(
        None, ge=1, le=5, description="Work–life balance strain (1–5)"
    )
    performance: Optional[float] = Field(
        None, ge=1, le=5, description="Performance evaluation pressure (1–5)"
    )
    time_pressure: Optional[float] = Field(
        None, ge=1, le=5, description="Time / target pressure (1–5)"
    )
    emotional_impact: Optional[float] = Field(
        None, ge=1, le=5, description="Emotional / burnout impact (1–5)"
    )
    financial: Optional[float] = Field(
        None, ge=1, le=5, description="Financial stress (1–5)"
    )
    social_support: Optional[float] = Field(
        None, ge=1, le=5, description="Lack of social support (1–5; higher = less support)"
    )

    tech_awareness: Optional[bool] = Field(
        None, description="Aware of tech-based mental health tools"
    )
    financial_literacy_training: Optional[bool] = Field(
        None, description="Received financial literacy training (Fin_D4)"
    )

    # Optional overrides of engineered column names
    feature_overrides: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Optional exact engineered feature overrides after expansion. "
            "Ignore Swagger additionalProp* placeholders."
        ),
        json_schema_extra={
            "example": {"Workload_A1": 5.0, "Fin_C5": 4.0},
        },
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age_category": "35-44",
                "gender": "Female",
                "years_in_service": "5-10 years",
                "school_type": "Primary School",
                "education_qualification": "Bachelor's Degree",
                "ict_skills": "Intermediate",
                "workload": 4.0,
                "learners": 4.0,
                "work_life_balance": 4.5,
                "performance": 3.5,
                "time_pressure": 4.0,
                "emotional_impact": 4.0,
                "financial": 3.5,
                "social_support": 3.0,
                "tech_awareness": False,
            }
        }
    }


class FullFeaturesInput(BaseModel):
    """
    Partial or full engineered feature map.

    Missing keys are filled with defaults (Likert=3, else 0) — never rejected.
    Note: Swagger may show "additionalProp1" for free-form dicts — replace those
    with real engineered names (e.g. Workload_A1). You do not need all ~222 keys.
    """

    features: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Engineered feature name → value. Any subset is OK. "
            "Ignore Swagger's additionalProp1/2/3 placeholders — use real names."
        ),
        json_schema_extra={
            "example": {
                "Workload_A1": 4.0,
                "Workload_A4": 5.0,
                "WLB_D2": 4.5,
                "Emot_C4": 4.0,
                "Fin_C5": 3.0,
                "Soc_D3": 3.5,
                "Gender_Female": 1.0,
                "Age_Category_35_44": 1.0,
                "School_Type_Primary_School": 1.0,
            }
        },
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": {
                    "Workload_A1": 4.0,
                    "Workload_A4": 5.0,
                    "WLB_D2": 4.5,
                    "Emot_C4": 4.0,
                    "Fin_C5": 3.0,
                    "Soc_D3": 3.5,
                    "Gender_Female": 1.0,
                    "Age_Category_35_44": 1.0,
                    "School_Type_Primary_School": 1.0,
                }
            }
        }
    }


class PredictionOutput(BaseModel):
    high_risk_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_risk: Literal["Low", "High"]
    risk_level: str
    confidence: Optional[str] = None
    features_used: int = Field(..., description="Length of full model feature vector")
    fields_provided: Optional[List[str]] = Field(
        None, description="Questionnaire fields / overrides the client supplied"
    )
    defaults_applied: bool = Field(
        True,
        description="True when any feature used a default (almost always for /predict)",
    )
    validation_passed: bool = Field(
        True,
        description="True if Great-Expectations-style / domain validation passed before predict",
    )
    validation_warnings: Optional[List[str]] = Field(
        None,
        description="Non-fatal validation warnings (e.g. unknown demographic label)",
    )
    disclaimer: str = Field(
        default=(
            "IMPORTANT DISCLAIMER: This is an AI-powered screening support tool only. "
            "It is NOT a substitute for professional medical or psychological diagnosis, "
            "treatment, or advice. A 'High Risk' result indicates elevated symptoms "
            "consistent with moderate-to-severe depression based on historical survey "
            "patterns and should prompt supportive follow-up by qualified mental health "
            "professionals or institutional support services. "
            "This tool must never be used for punitive, disciplinary, or employment "
            "decisions. All predictions require human oversight and clinical validation."
        )
    )
    model_version: str = "mlflow-or-local"


class ValidationErrorResponse(BaseModel):
    """HTTP 400 body when input validation fails."""

    detail: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    feature_count: int
    version: str = "week1-2-mlflow"
    model_source: Optional[str] = None
    model_name: Optional[str] = None


# Backward-compatible alias
PredictionInput = FullFeaturesInput
