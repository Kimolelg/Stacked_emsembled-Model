import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { listFeastTeachers, predict } from "../api/client";
import type { PredictPayload, PredictionResult } from "../types";

const AGE_OPTIONS = ["Below 25", "25-34", "35-44", "45-54", "55 and above"];
const GENDER_OPTIONS = ["Female", "Male", "Other"];
const YEARS_OPTIONS = [
  "Less than 5 years",
  "5-10 years",
  "11-15 years",
  "16-20 years",
  "Over 20 years",
];
const SCHOOL_OPTIONS = ["Primary School", "Secondary School", "Junior School"];
const EDU_OPTIONS = ["Certificate/Diploma", "Bachelor's Degree", "Master's Degree"];
const ICT_OPTIONS = ["Basic", "Intermediate", "Advanced"];

const SECTION_FIELDS: Array<{ key: keyof PredictPayload; label: string; help: string }> = [
  { key: "workload", label: "Workload demand", help: "Teaching and non-teaching load" },
  { key: "learners", label: "Class load", help: "Pressure from class size" },
  { key: "work_life_balance", label: "Work–life balance", help: "Recovery and personal time" },
  { key: "performance", label: "Performance pressure", help: "Evaluation and targets" },
  { key: "time_pressure", label: "Time pressure", help: "Deadlines and pacing" },
  { key: "emotional_impact", label: "Emotional strain", help: "Exhaustion and burnout feelings" },
  { key: "financial", label: "Financial stress", help: "Money worries" },
  { key: "social_support", label: "Limited social support", help: "Higher means less support" },
];

const emptyForm: PredictPayload = {
  age_category: "25-34",
  gender: "Female",
  years_in_service: "5-10 years",
  school_type: "Primary School",
  education_qualification: "Bachelor's Degree",
  ict_skills: "Basic",
  workload: 3,
  learners: 3,
  work_life_balance: 3,
  performance: 3,
  time_pressure: 3,
  emotional_impact: 3,
  financial: 3,
  social_support: 3,
  tech_awareness: false,
  teacher_id: null,
};

export default function ScreeningPage() {
  const [mode, setMode] = useState<"questionnaire" | "directory">("questionnaire");
  const [form, setForm] = useState<PredictPayload>(emptyForm);
  const [teacherId, setTeacherId] = useState("");
  const [sampleIds, setSampleIds] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  useEffect(() => {
    listFeastTeachers(15)
      .then((data) => {
        const ids = data.teacher_ids || data.teachers?.map((t) => t.teacher_id) || [];
        setSampleIds(ids.filter((x) => typeof x === "number"));
      })
      .catch(() => setSampleIds([]));
  }, []);

  function setField<K extends keyof PredictPayload>(key: K, value: PredictPayload[K]) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      let payload: PredictPayload;
      if (mode === "directory") {
        const id = Number(teacherId);
        if (!Number.isFinite(id) || id < 1) {
          throw new Error("Enter a valid teacher number (1 or higher).");
        }
        payload = { teacher_id: id };
      } else {
        payload = { ...form, teacher_id: null };
      }
      setResult(await predict(payload));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const pct = result
    ? Number((result.high_risk_probability * 100).toFixed(1))
    : 0;

  return (
    <div className="page">
      <section className="page-header">
        <div>
          <p className="eyebrow">Wellness screening</p>
          <h2>Assess wellbeing risk</h2>
          <p className="lead">
            Complete a short questionnaire or look up a saved teacher profile. Results
            support confidential outreach planning.
          </p>
        </div>
        <ul className="trust-chips" aria-label="Usage principles">
          <li>Confidential use</li>
          <li>Human follow-up</li>
          <li>Not a diagnosis</li>
        </ul>
      </section>

      <div className="mode-toggle" role="tablist" aria-label="Input mode">
        <button
          type="button"
          className={mode === "questionnaire" ? "active" : ""}
          onClick={() => setMode("questionnaire")}
        >
          New questionnaire
        </button>
        <button
          type="button"
          className={mode === "directory" ? "active" : ""}
          onClick={() => setMode("directory")}
        >
          Saved teacher profile
        </button>
      </div>

      <form className="card form panel" onSubmit={onSubmit}>
        {mode === "directory" ? (
          <div className="field">
            <label htmlFor="teacher_id">Teacher number</label>
            <input
              id="teacher_id"
              type="number"
              min={1}
              value={teacherId}
              onChange={(e) => setTeacherId(e.target.value)}
              placeholder="Enter teacher number"
              required
            />
            {sampleIds.length > 0 && (
              <div className="chip-picks" aria-label="Available profiles">
                {sampleIds.slice(0, 8).map((id) => (
                  <button
                    key={id}
                    type="button"
                    className={teacherId === String(id) ? "chip active" : "chip"}
                    onClick={() => setTeacherId(String(id))}
                  >
                    #{id}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <>
            <h3 className="section-title">Background</h3>
            <div className="grid-2">
              <div className="field">
                <label>Age category</label>
                <select
                  value={form.age_category || ""}
                  onChange={(e) => setField("age_category", e.target.value)}
                >
                  {AGE_OPTIONS.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Gender</label>
                <select
                  value={form.gender || ""}
                  onChange={(e) => setField("gender", e.target.value)}
                >
                  {GENDER_OPTIONS.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Years in service</label>
                <select
                  value={form.years_in_service || ""}
                  onChange={(e) => setField("years_in_service", e.target.value)}
                >
                  {YEARS_OPTIONS.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>School type</label>
                <select
                  value={form.school_type || ""}
                  onChange={(e) => setField("school_type", e.target.value)}
                >
                  {SCHOOL_OPTIONS.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Education</label>
                <select
                  value={form.education_qualification || ""}
                  onChange={(e) => setField("education_qualification", e.target.value)}
                >
                  {EDU_OPTIONS.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>Digital skills</label>
                <select
                  value={form.ict_skills || ""}
                  onChange={(e) => setField("ict_skills", e.target.value)}
                >
                  {ICT_OPTIONS.map((o) => (
                    <option key={o} value={o}>{o}</option>
                  ))}
                </select>
              </div>
            </div>

            <h3 className="section-title">Wellbeing pressures (1 low → 5 high)</h3>
            <div className="slider-grid">
              {SECTION_FIELDS.map(({ key, label, help }) => (
                <div className="slider-card" key={String(key)}>
                  <div className="slider-head">
                    <label htmlFor={String(key)}>{label}</label>
                    <span className="slider-value">{String(form[key] ?? 3)}</span>
                  </div>
                  <input
                    id={String(key)}
                    type="range"
                    min={1}
                    max={5}
                    step={0.5}
                    value={Number(form[key] ?? 3)}
                    onChange={(e) => setField(key, Number(e.target.value) as never)}
                  />
                  <span className="hint">{help}</span>
                </div>
              ))}
            </div>

            <label className="checkbox">
              <input
                type="checkbox"
                checked={Boolean(form.tech_awareness)}
                onChange={(e) => setField("tech_awareness", e.target.checked)}
              />
              Aware of digital mental health tools
            </label>
          </>
        )}

        <div className="btn-row sticky-actions">
          <button className="primary" type="submit" disabled={loading}>
            {loading ? (
              <>
                <span className="spinner" /> Assessing…
              </>
            ) : (
              "Run screening"
            )}
          </button>
          <button
            type="button"
            className="ghost"
            onClick={() => {
              setForm(emptyForm);
              setResult(null);
              setError(null);
              setTeacherId("");
            }}
          >
            Clear
          </button>
        </div>
      </form>

      {error && (
        <div className="card error" role="alert">
          <strong>Screening could not be completed</strong>
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className={`card result ${result.predicted_risk === "High" ? "high" : "low"}`}>
          <div className="result-header">
            <div>
              <p className="eyebrow">Screening result</p>
              <h3>
                {result.predicted_risk === "High"
                  ? "Elevated support need"
                  : "Lower support need"}
              </h3>
            </div>
            <div className="score-ring" aria-label={`${pct} percent elevated risk`}>
              <div className="pct">{pct}%</div>
            </div>
          </div>

          <p className="result-summary">
            <strong>{result.predicted_risk}</strong>
            {result.confidence ? ` · ${result.confidence} confidence` : ""}
            {result.risk_level ? ` · ${result.risk_level}` : ""}
          </p>

          {result.phq9_estimated != null && (
            <div className="phq-box">
              <div className="phq-head">
                <span>Estimated wellbeing score</span>
                <strong>
                  {result.phq9_estimated}
                  <span className="hint"> / 27</span>
                </strong>
              </div>
              {result.phq9_interpretation && (
                <p className="phq-interp">{result.phq9_interpretation}</p>
              )}
              <div className="phq-bar" aria-hidden>
                <div
                  className="phq-fill"
                  style={{ width: `${(result.phq9_estimated / 27) * 100}%` }}
                />
              </div>
            </div>
          )}

          {result.validation_warnings && result.validation_warnings.length > 0 && (
            <ul className="warnings">
              {result.validation_warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}

          {result.disclaimer && <p className="disclaimer">{result.disclaimer}</p>}
        </div>
      )}
    </div>
  );
}
