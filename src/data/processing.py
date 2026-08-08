"""
Mental Health Risk Prediction - Data Processing Pipeline
Week 1-2 Deliverable

This module encapsulates all data cleaning, feature engineering,
and target creation logic from the validated development notebook
(`mental_health_final_model.py`).

It is designed to be:
- Reproducible
- Testable
- Suitable for both training and inference pipelines
- Faithful to the notebook pipeline (order of operations preserved)
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

# Multicollinear / low-value columns removed after EDA in the final model notebook
MULTICOLLINEAR_DROP = [
    "Perf_A2",
    "Perf_A5",
    "Time_B5",
    "Time_B2",
    "Emot_C2",
    "Emot_C5",
    "Emot_C1",
    "Learners_B4",
    "County",
]

# Self-reported diagnosis / clinical history → label leakage for early screening
LEAKAGE_COLS = [
    "Depression_Experience",
    "Depression_Types_None_of_the_above",
    "Depression_Types_Secondary_Comorbid_Depression__Linked_to_another_health_or_psychological_condition_",
    "Depression_Types_Situational_Depression___Triggered_by_specific_events_or_stressors_at_work_",
    "Depression_Types_Atypical_Depression_Mood_improves_with_positive_events__increased_sleep_eating_",
    "Depression_Types_Seasonal_Affective_Disorder__Symptoms_worsen_during_particular_seasons_",
    "Depression_Types_Dysthymia__Chronic_low_mood_over_a_long_period_",
    "Depression_Types_Burnout_Related_Depression__Due_to_prolonged_work_related_stress_or_exhaustion_",
    "Depression_Types__Major_Depressive_Disorder__Persistent_sadness__lack_of_pleasure_not_limited_to_work_",
]

# Demographics / yes-no handled as categoricals (matches notebook)
CATEGORICAL_COLS = [
    "Age_Category",
    "Gender",
    "Years_in_Service",
    "School_Type",
    "Sub_County",
    "Education_Qualification",
    "ICT_Skills",
    "Depression_Experience",
    "Tech_Awareness",
    "Fin_D4",
]

# One-hot encode all categoricals except the last three binary yes/no fields
# notebook: pd.get_dummies(df, columns=categorical_cols[:-3])
OHE_COLS = CATEGORICAL_COLS[:-3]

BINARY_COLS = ["Fin_D4", "Tech_Awareness", "Depression_Experience"]

PHQ_COLS = ["PHQ1", "PHQ2", "PHQ3", "PHQ4", "PHQ5", "PHQ6", "PHQ7", "PHQ8", "PHQ9"]

LIKERT_MAP = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
    "Never": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
    "Always": 5,
    "Most of the time": 4,
}

PHQ_MAP = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the day": 2,
    "Nearly every day": 3,
}


def clean_column_name(col: str) -> str:
    """Replace any non-alphanumeric character (except underscore) with underscore."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", col)


def get_column_rename_map() -> dict:
    """
    Full survey question → short feature name map from the validated notebook.
    Includes deployment/ethics items (Dep_*) so they can be dropped cleanly after rename.
    """
    return {
        "What is your Age category: (Please tick (√) your response)?": "Age_Category",
        "What is your Gender: (Please tick (√) your response)?": "Gender",
        "How many Years have you been in the Teaching Service?      (Please tick (√) your response)?": "Years_in_Service",
        "Which Type of Public School are you in? (Please tick (√) your response)?": "School_Type",
        "What is the name of your work Sub-County?": "Sub_County",
        "What is the name of your County?": "County",
        "What is your level Educational Qualification? ": "Education_Qualification",
        "How would you rate your current level of ICT skills in performing teaching and administrative tasks?": "ICT_Skills",
        "Have you experienced any signs or symptoms of depression for example: persistent sadness, loss of interest in activities, fatigue, difficulty concentrating, changes in sleep or appetite) in the co...": "Depression_Experience",
        "If yes, what form of depression have you experience. (You may select more than one if applicable)": "Depression_Types",
        "Are you aware of any technology-based solutions e.g. (Machine Learning Models, Mobile Apps, Wearable Devices, Chatbots or Self-assessment Platforms) designed to help manage depression?": "Tech_Awareness",
        "If yes, Select the platform you have used you may select more than one if applicable.If not applicable, indicate “Not applicable”": "Tech_Platforms",
        # Job Demands - Workload
        "(a). I have more classroom teaching hours than I can comfortably manage.": "Workload_A1",
        "(b). My non-teaching responsibilities increase my overall workload.": "Workload_A2",
        "(c). I handle more subjects than I can comfortably manage": "Workload_A3",
        "        (d). I often have to prepare for different lessons on the same   day.      ": "Workload_A4",
        "(e). The amount of work grading and assessment is overwhelming": "Workload_A5",
        # Learners Load
        "(a). The number of learners in my class(es) makes teaching difficult.": "Learners_B1",
        "(b). Large class sizes prevent me from giving adequate attention to each learner": "Learners_B2",
        "        (c). I struggle to complete lesson planning due to the number   of students I teach      ": "Learners_B3",
        "(d). I often carry over learners marking and preparation of reports into my personal time.": "Learners_B4",
        # Work-Life Balance
        "(a). My workload negatively affects my personal life.": "WLB_D1",
        "        (b). I frequently feel mentally exhausted due to my work   responsibilities": "WLB_D2",
        "(c). I do not have enough time to rest and recover between school days. ": "WLB_D3",
        "(d). I feel that my current workload is not sustainable in the long term.": "WLB_D4",
        # Performance Evaluation
        "(a). My promotion depends heavily on my performance evaluation results": "Perf_A1",
        "        (b). The performance evaluation process in my   school is fair and transparent.       ": "Perf_A2",
        "(c). I often worry about not being promoted due to low performance ratings.": "Perf_A3",
        "(d). My performance is assessed based on standards beyond my control ": "Perf_A4",
        "(e). My workload has increased due to pressure to meet performance targets.": "Perf_A5",
        # Time Pressure
        "(a). I am expected to meet unrealistic teaching and performance targets.": "Time_B1",
        "(b). I frequently take work home to complete after school hours.": "Time_B2",
        "(c). My performance evaluation does not consider my workload and resource limitations.": "Time_B3",
        "(d). The pressure to achieve high student outcomes negatively affects my mental well-being.": "Time_B4",
        "(e). I feel constantly exhausted due to performance-related stress.": "Time_B5",
        # Emotional Impact
        "        (a). I often feel anxious before performance   evaluations.      ": "Emot_C1",
        "(b). Negative feedback from performance reviews affects my confidence and motivation.": "Emot_C2",
        "(c). The pressure to meet performance targets has contributed to burnout.": "Emot_C3",
        "(d). My mental well-being has declined due to performance-related stress.": "Emot_C4",
        "(e). I have considered leaving the teaching profession due to performance pressures.": "Emot_C5",
        # Financial
        "My salary is sufficient to cover my essential living expenses (e.g., rent, food, transportation).": "Fin_A1",
        "I frequently run out of money before the next paycheck.": "Fin_A2",
        "My salary is paid on time without delays.": "Fin_A3",
        "Over the past year, my salary has increased in line with inflation or cost of living.": "Fin_A4",
        " I have considered leaving the teaching profession due to financial instability.": "Fin_B2",
        " I rely on additional jobs, coaching, or side businesses to supplement my income.": "Fin_B3",
        "My workload exceeds what I am paid for.": "Fin_B4",
        "I often borrow money from family, friends, or financial institutions to meet basic needs.": "Fin_C2",
        "I experience stress or depression due to my financial situation.": "Fin_C3",
        " Financial stress has negatively affected my mental health.": "Fin_C4",
        "I have skipped meals or essential purchases due to financial constraints.": "Fin_C5",
        " I have a retirement plan or pension that will provide sufficient financial security after I stop working.": "Fin_D2",
        "I feel confident about my long-term financial future.": "Fin_D3",
        "I have received financial literacy training or education on managing personal finances.": "Fin_D4",
        # Social Support
        "(a). I feel isolated from my colleagues at school.": "Soc_A1",
        "(b). My colleagues rarely offer help when I am overwhelmed with work.": "Soc_A2",
        "(c). I do not have anyone at school to share my challenges with.": "Soc_A3",
        "        (d). There is a lack of teamwork among teachers in   my school.      ": "Soc_A4",
        "(e). I do not receive emotional support from my colleagues/HOS/HOI.": "Soc_A5",
        "        (a). My school Head of Institution does not   recognize or appreciate my efforts.      ": "Soc_B1",
        "(b). My Head of subject (HOS) do not provide guidance or support when I am struggling": "Soc_B2",
        "        (c). I feel that my concerns and grievances are   ignored by the school administration.      ": "Soc_B3",
        "(d). I do not receive constructive feedback that helps me improve my teaching.": "Soc_B4",
        "(e). My workload is not taken into consideration when assigning new tasks.": "Soc_B5",
        "(a). Parents do not support me in managing student behavior and discipline.": "Soc_C1",
        "        (b). I frequently feel blamed by parents for their   children's poor academic performance.      ": "Soc_C2",
        "        (c). I receive little or no recognition from the   community for my work as a teacher.      ": "Soc_C3",
        "        (d). There is minimal parental involvement in their   children's education at my school.      ": "Soc_C4",
        "(e). I feel disconnected from the broader community in which I teach.": "Soc_C5",
        "(a). My school does not offer mental health resources or counselling services.": "Soc_D1",
        "(b). I feel discouraged from seeking help for mental health concerns.": "Soc_D2",
        "(c). There is stigma around discussing Depression and well-being among teachers.": "Soc_D3",
        "(d). I feel unsupported when dealing with emotional distress related to my job.": "Soc_D4",
        "(e). I do not have anyone to turn to when I feel overwhelmed or stressed.": "Soc_D5",
        # Deployment Strategies (dropped after rename — not model features)
        "        (a). I am aware of   predictive model being used in education to monitor mental health      ": "Dep_A1",
        "        (b). A predictive model   for teacher depression would be valuable in our school       ": "Dep_A2",
        "(c). My school is currently facing challenges in identifying teacher mental health needs": "Dep_A3",
        "(d). Using a predictive model would improve early detection and support for affected teachers": "Dep_A4",
        "        (e). Data driven tools can   enhance mental health decisions in our school      ": "Dep_A5",
        "(a). Our school has the technological infrastructure required to access a predictive model": "Dep_B1",
        "        (b). Teachers and   administrators would need structured training to effectively use the model      ": "Dep_B2",
        "(c). With basic training most teachers can use a mental health prediction system effectively": "Dep_B3",
        "(d). Integrating a predictive model into existing TSC systems would be technically possible": "Dep_B4",
        "(e). A user-friendly interface is important for successful deployment of such a model ": "Dep_B5",
        "(a). I would support the implementation of a predictive model in our school.": "Dep_C1",
        "(b). School leadership is likely to support technology driven mental health initiatives.": "Dep_C2",
        "        (c). Stakeholders (Unions)   would support an evidence-based approach in addressing teacher depression      ": "Dep_C3",
        "(d). Government, MOE & TSC support is essential for successful deployment and usage of the model": "Dep_C4",
        "(a). Data privacy concerns could limit model adoption": "Dep_D1",
        "(b). Resistance to new technology is a common challenge in our school.": "Dep_D2",
        "(c). Limited ICT resources could affect the usability of the model": "Dep_D3",
        "(d). Funding constraints may hinder the implementation of the predictive model": "Dep_D4",
        "(e). Mental health stigma may reduce willingness to use such a tool": "Dep_D5",
        "        (a). I am concerned about   how my data would be handled in a predictive model.      ": "Dep_E1",
        "(b). Teachers’ personal data should be anonymized before being used in any model": "Dep_E2",
        "(c). Only authorized personnel should have access to sensitive mental health data": "Dep_E3",
        "(d). Consent must be obtained before collecting data for mental health prediction": "Dep_E4",
        "(e). Secure data storage systems should be a prerequisite for model deployment": "Dep_E5",
        # PHQ-9
        "(a). Little interest or pleasure in doing things": "PHQ1",
        "(b). Feeling down, depressed, or hopeless": "PHQ2",
        "(c). Trouble falling or staying asleep, or sleeping too much": "PHQ3",
        "(d). Feeling tired or   having little energy      ": "PHQ4",
        "        (e). Poor appetite or   overeating      ": "PHQ5",
        "(f). Feeling bad about yourself — or that you are a failure or have let yourself or your family down": "PHQ6",
        "(g). Trouble concentrating on things, such as reading the newspaper or watching the TV": "PHQ7",
        "(h). Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual": "PHQ8",
        "(i). Thoughts that you would be better off dead or of hurting yourself in some way": "PHQ9",
    }


def load_and_clean_raw_data(
    file_path: str,
    sheet_name: str = "Sheet1",
) -> pd.DataFrame:
    """
    Load the raw TSC teacher survey Excel file and perform initial cleaning.

    Matches notebook steps:
    - Load Excel
    - Drop non-informative columns (timestamps, consent, IDs, contact fields)
    - Rename long survey questions to short feature names
    - Drop deployment / ethics awareness items (Dep_*) and ID
    """
    logger.info(f"Loading raw data from {file_path}")

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    logger.info(f"Initial shape: {df.shape}")

    # Notebook columns_to_drop (include common Excel export variants)
    columns_to_drop = [
        "Start time",
        "Completion time",
        "Email",
        "Name",
        "Last modified time",
        "Mobile Number:",
        "Mobile Number:\xa0",
        "Date:",
        "Date:\xa0",
        "Thank You",
        "Consent Statement",
        # Full consent header as exported from Forms (prefix match handled below)
    ]
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

    # Drop any remaining consent-like columns by prefix (export text varies)
    consent_cols = [
        c
        for c in df.columns
        if isinstance(c, str) and c.startswith("Consent Statement")
    ]
    if consent_cols:
        df = df.drop(columns=consent_cols, errors="ignore")

    df = df.rename(columns=get_column_rename_map())

    # Drop deployment/ethics awareness questions + ID (notebook list)
    dep_and_id = [
        "ID",
        "Dep_A1",
        "Dep_A2",
        "Dep_A3",
        "Dep_A4",
        "Dep_A5",
        "Dep_B1",
        "Dep_B2",
        "Dep_B3",
        "Dep_B4",
        "Dep_B5",
        "Dep_C1",
        "Dep_C2",
        "Dep_C3",
        "Dep_C4",
        "Dep_D1",
        "Dep_D2",
        "Dep_D3",
        "Dep_D4",
        "Dep_D5",
        "Dep_E1",
        "Dep_E2",
        "Dep_E3",
        "Dep_E4",
        "Dep_E5",
    ]
    # Also catch any Dep_* that survived rename mismatches
    dep_and_id = list(
        dict.fromkeys(
            dep_and_id + [c for c in df.columns if str(c).startswith("Dep_")]
        )
    )
    df = df.drop(columns=[c for c in dep_and_id if c in df.columns], errors="ignore")

    logger.info(f"After rename and Dep_/ID drop: {df.shape}")
    return df


def _severity_label(score: float) -> str:
    """PHQ-9 severity band used in notebook EDA (not a model feature)."""
    if score <= 9:
        return "Low"  # Minimal + Mild
    if score <= 14:
        return "Moderate"
    return "High"  # Moderately Severe + Severe


def engineer_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering and target creation pipeline.

    Order of operations mirrors mental_health_final_model.py:
    1. Mode-impute categoricals
    2. Multi-select one-hot (Depression_Types, Tech_Platforms)
    3. Likert map (exclude Fin_D4 — it is Yes/No binary)
    4. PHQ-9 map → PHQ_Total → drop item-level PHQ columns
    5. One-hot encode demographics including Sub_County
    6. Binary Yes/No encoding
    7. Drop multicollinear columns (EDA)
    8. Sanitize column names
    9. Drop residual object columns
    10. Create Depression_Severity (EDA) + High_Risk target
    11. Remove label-leakage diagnosis columns (AFTER multi-select dummies exist)
    """
    logger.info("Starting feature engineering and target creation...")
    df = df.copy()

    # --- 1. Mode impute categoricals (notebook: SimpleImputer most_frequent) ---
    existing_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    if existing_cats:
        for col in existing_cats:
            mode_vals = df[col].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if len(mode_vals) else "Unknown"
            df[col] = df[col].fillna(fill_value)
        logger.info(f"Mode-imputed categoricals: {existing_cats}")

    # --- 2. Multi-select → one-hot (must happen BEFORE leakage drop) ---
    multi_select_cols = ["Depression_Types", "Tech_Platforms"]
    for col in multi_select_cols:
        if col in df.columns:
            df[col] = df[col].fillna("None")
            dummies = df[col].str.get_dummies(sep=";")
            dummies.columns = [
                f"{col}_{sub.replace(' ', '_')}" for sub in dummies.columns
            ]
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)

    # --- 3. Likert ordinal mapping (exclude Fin_D4 — Yes/No, not Likert) ---
    ordinal_likert_cols = [
        col
        for col in df.columns
        if col.startswith(
            ("Workload_", "Learners_", "WLB_", "Perf_", "Time_", "Emot_", "Fin_", "Soc_")
        )
        and col not in ("Fin_D4",)
    ]
    for col in ordinal_likert_cols:
        df[col] = df[col].map(LIKERT_MAP).fillna(3)

    # --- 4. PHQ-9 items → numeric, total, then drop items as features ---
    present_phq = [c for c in PHQ_COLS if c in df.columns]
    for col in present_phq:
        df[col] = df[col].map(PHQ_MAP).fillna(0)

    if present_phq:
        df["PHQ_Total"] = df[present_phq].sum(axis=1)
        df = df.drop(columns=present_phq)
    else:
        logger.warning("No PHQ columns found; PHQ_Total cannot be computed.")
        df["PHQ_Total"] = 0.0

    # --- 5. One-hot encode demographics (includes Sub_County) ---
    ohe_cols = [c for c in OHE_COLS if c in df.columns]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)
        logger.info(f"One-hot encoded: {ohe_cols}")

    # --- 6. Binary Yes/No (force float so they survive the object-column cleanup) ---
    yes_no_map = {"Yes": 1.0, "No": 0.0, "yes": 1.0, "no": 0.0}
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(lambda v: yes_no_map.get(v, v))
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                mode_vals = df[col].mode(dropna=True)
                fill = float(mode_vals.iloc[0]) if len(mode_vals) else 0.0
                df[col] = df[col].fillna(fill)
            df[col] = df[col].astype(float)

    # --- 7. Multicollinearity drops from notebook EDA ---
    drop_mc = [c for c in MULTICOLLINEAR_DROP if c in df.columns]
    if drop_mc:
        df = df.drop(columns=drop_mc)
        logger.info(f"Dropped multicollinear columns: {drop_mc}")

    # --- 8. Sanitize all column names for model compatibility ---
    df.columns = [clean_column_name(col) for col in df.columns]

    # --- 9. Drop remaining object columns (unmapped strings only) ---
    # Protect binary / target-related numeric columns from accidental drop
    protect = set(BINARY_COLS) | {
        clean_column_name(c) for c in BINARY_COLS
    } | {"PHQ_Total", "High_Risk", "Depression_Severity"}
    object_cols = [
        c
        for c in df.select_dtypes(include="object").columns.tolist()
        if c not in protect
    ]
    if object_cols:
        logger.warning(f"Dropping remaining object columns: {object_cols}")
        df = df.drop(columns=object_cols)

    # --- 10. Targets ---
    df["Depression_Severity"] = df["PHQ_Total"].apply(_severity_label)
    df["High_Risk"] = (df["PHQ_Total"] >= 15).astype(int)

    # --- 11. Remove label-leakage columns AFTER dummies + clean_column_name ---
    # Clean names on leakage list for safety (notebook already uses cleaned forms)
    leakage_cleaned = [clean_column_name(c) for c in LEAKAGE_COLS]
    # Also drop any Depression_Types_* dummy that may appear with slight name variants
    depression_type_dummies = [
        c for c in df.columns if c.startswith("Depression_Types_")
    ]
    to_remove = list(
        dict.fromkeys(
            [c for c in leakage_cleaned if c in df.columns] + depression_type_dummies
        )
    )
    if to_remove:
        df = df.drop(columns=to_remove)
        logger.info(f"Removed label-leakage columns ({len(to_remove)}): {to_remove}")

    # Ensure modeling columns are numeric (Depression_Severity stays object until prepare)
    numeric_ready = [
        c for c in df.columns if c not in ("Depression_Severity",)
    ]
    for col in numeric_ready:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Any residual NaNs from coerce → neutral/0-safe fill for tree models
    likert_like = [
        c
        for c in df.columns
        if c.startswith(
            ("Workload_", "Learners_", "WLB_", "Perf_", "Time_", "Emot_", "Fin_", "Soc_")
        )
        and c != "Fin_D4"
    ]
    for col in likert_like:
        if col in df.columns:
            df[col] = df[col].fillna(3.0)

    other_num = [
        c
        for c in df.columns
        if c not in likert_like and c not in ("Depression_Severity",)
    ]
    for col in other_num:
        df[col] = df[col].fillna(0.0)

    logger.info(f"Final engineered shape: {df.shape}")
    logger.info(f"High Risk prevalence: {df['High_Risk'].mean():.1%}")

    return df


def get_feature_order(df: pd.DataFrame) -> List[str]:
    """Return the exact ordered list of feature columns (excluding targets / labels)."""
    exclude = ["PHQ_Total", "High_Risk", "Depression_Severity"]
    return [col for col in df.columns if col not in exclude]


def prepare_for_modeling(
    df: pd.DataFrame,
    target_col: str = "High_Risk",
) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """
    Final preparation step: returns X, y, and feature_order.

    Mirrors notebook:
        X = df.drop(columns=['PHQ_Total', 'Depression_Severity', 'High_Risk'] + leakage)
        X = X.astype(float)
        y = df['High_Risk']
    """
    feature_order = get_feature_order(df)
    X = df[feature_order].copy().astype(float)
    y = df[target_col].copy() if target_col in df.columns else None
    return X, y, feature_order


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Data processing module loaded successfully.")
    print("Pipeline aligned with mental_health_final_model.py (Week 1-2).")
