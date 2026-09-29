export type PredictPayload = {
  teacher_id?: number | null;
  age_category?: string | null;
  gender?: string | null;
  years_in_service?: string | null;
  school_type?: string | null;
  education_qualification?: string | null;
  ict_skills?: string | null;
  workload?: number | null;
  learners?: number | null;
  work_life_balance?: number | null;
  performance?: number | null;
  time_pressure?: number | null;
  emotional_impact?: number | null;
  financial?: number | null;
  social_support?: number | null;
  tech_awareness?: boolean | null;
};

export type PredictionResult = {
  high_risk_probability: number;
  predicted_risk: "Low" | "High";
  risk_level: string;
  confidence?: string | null;
  phq9_estimated?: number | null;
  phq9_severity?: string | null;
  phq9_interpretation?: string | null;
  phq9_note?: string | null;
  features_used: number;
  fields_provided?: string[] | null;
  defaults_applied: boolean;
  validation_passed: boolean;
  validation_warnings?: string[] | null;
  disclaimer: string;
  model_version?: string;
};

export type HealthResponse = {
  status: string;
  model_loaded: boolean;
  feature_count: number;
  version: string;
  model_source: string;
  model_name: string;
};

export type MonitoringStatus = {
  production_log_rows: number;
  production_log_path?: string;
  reference_exists: boolean;
  reference_path?: string;
  last_drift_summary?: DriftSummary | Record<string, unknown> | null;
  min_rows_for_drift?: number;
  hint?: string;
};

export type DriftSummary = {
  success?: boolean;
  timestamp?: string;
  n_reference?: number;
  n_current?: number;
  dataset_drift?: boolean;
  share_drifted?: number;
  drift_share_threshold?: number;
  alert?: boolean;
  alert_message?: string;
  feature_columns?: string[];
  evidently?: {
    engine?: string;
    report_path?: string;
    error?: string | null;
  };
  mean_shifts?: Record<
    string,
    { reference_mean?: number; current_mean?: number; delta?: number }
  >;
};

export type EvidentlySchedule = {
  enabled: boolean;
  interval_hours: number;
  build_reference?: boolean;
  min_current_rows?: number;
  next_run_utc?: string | null;
  last_run_utc?: string | null;
  last_status?: string | null;
  last_error?: string | null;
  last_summary?: Record<string, unknown> | null;
};

export type FeastVersionsResponse = {
  current?: { version?: string; uploaded_at_utc?: string } | null;
  versions: Array<Record<string, unknown>>;
  count: number;
};

export type ApiErrorBody = {
  detail?:
    | string
    | {
        message?: string;
        errors?: string[];
        warnings?: string[];
      };
};
