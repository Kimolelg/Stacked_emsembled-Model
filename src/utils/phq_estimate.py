"""Map High_Risk probability to an approximate PHQ-9 total (0–27) and severity band."""

from __future__ import annotations

from typing import Dict, Tuple


def estimate_phq9_from_probability(proba: float) -> Dict[str, object]:
    """
    Calibrated estimate: P(High_Risk)=0.5 ≈ PHQ-9 total 15 (screening cut-off).

    phq ≈ clip(15 + 12*(2p - 1), 0, 27)
      p=0 → ~3, p=0.5 → 15, p=1 → 27
    """
    p = float(max(0.0, min(1.0, proba)))
    raw = 15.0 + 12.0 * (2.0 * p - 1.0)
    score = int(round(max(0.0, min(27.0, raw))))
    band, meaning = _severity_band(score)
    return {
        "phq9_estimated": score,
        "phq9_scale": "0-27",
        "phq9_severity": band,
        "phq9_interpretation": meaning,
        "phq9_note": (
            "Estimated PHQ-9 total derived from model High_Risk probability "
            "(calibrated so ~0.5 ≈ score 15). Not a completed PHQ-9 questionnaire."
        ),
    }


def _severity_band(score: int) -> Tuple[str, str]:
    if score <= 4:
        return "minimal", "Minimal symptoms (0–4)"
    if score <= 9:
        return "mild", "Mild symptoms (5–9)"
    if score <= 14:
        return "moderate", "Moderate symptoms (10–14)"
    if score <= 19:
        return "moderately_severe", "Moderately severe symptoms (15–19)"
    return "severe", "Severe symptoms (20–27)"
