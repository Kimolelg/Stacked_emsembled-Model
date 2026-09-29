import { NavLink, Outlet } from "react-router-dom";
import { useEffect, useState } from "react";
import { API_BASE, getHealth } from "../api/client";
import type { HealthResponse } from "../types";

type ModelInfoLite = {
  model_source?: string;
  model_name?: string;
  registered_model?: string;
};

function friendlyModelLabel(raw?: string | null): string {
  if (!raw) return "Ready";
  return raw
    .replace(/^teacher-mental-health-risk-/, "")
    .replace(/@champion$/i, "")
    .replace(/_/g, " ")
    .trim() || "Ready";
}

export default function Layout() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoLite | null>(null);
  const [healthErr, setHealthErr] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then((h) => {
        setHealth(h);
        setHealthErr(null);
      })
      .catch((e: Error) => setHealthErr(e.message));

    fetch(`${API_BASE}/model-info`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => j && setModelInfo(j))
      .catch(() => undefined);
  }, []);

  const rawName = modelInfo?.model_name || health?.model_name;
  const servingLabel = friendlyModelLabel(rawName);
  const showModelName =
    Boolean(servingLabel) &&
    servingLabel.toLowerCase() !== "ready" &&
    !/\.joblib$/i.test(rawName || "");

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden>
            <span />
          </div>
          <div>
            <h1>Teacher Wellness</h1>
            <p className="subtitle">Confidential screening for supportive outreach</p>
          </div>
        </div>
        <nav className="nav" aria-label="Primary">
          <NavLink to="/" end>
            Screening
          </NavLink>
          <NavLink to="/monitoring">Insights</NavLink>
        </nav>
        <div className="health-pill">
          {healthErr ? (
            <span className="badge bad">Unavailable</span>
          ) : health?.model_loaded ? (
            <span className="badge ok">
              {showModelName ? `Approved · ${servingLabel}` : "Approved model"}
            </span>
          ) : health ? (
            <span className="badge warn">Model unavailable</span>
          ) : (
            <span className="badge">Connecting…</span>
          )}
        </div>
      </header>

      <main className="main">
        <Outlet />
      </main>

      <footer className="footer">
        <p>
          For authorised wellness support only. Results guide confidential follow-up —
          they are not a medical diagnosis and must not inform discipline, hiring, or
          insurance decisions.
        </p>
      </footer>
    </div>
  );
}
