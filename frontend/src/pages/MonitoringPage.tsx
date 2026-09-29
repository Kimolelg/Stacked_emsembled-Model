import { useCallback, useEffect, useState } from "react";
import {
  buildReference,
  clearEvidentlySchedule,
  fetchMonitoringReportHtml,
  getEvidentlySchedule,
  getMonitoringStatus,
  listFeastVersions,
  runDrift,
  setEvidentlySchedule,
  uploadFeastFile,
} from "../api/client";
import type {
  DriftSummary,
  EvidentlySchedule,
  FeastVersionsResponse,
  MonitoringStatus,
} from "../types";

function asSummary(
  raw: MonitoringStatus["last_drift_summary"],
): DriftSummary | null {
  if (!raw || typeof raw !== "object") return null;
  return raw as DriftSummary;
}

function friendlyAlertMessage(raw?: string | null): string {
  const fallback =
    "Recent answers look different from the baseline. Review the report and latest survey uploads.";
  if (!raw) return fallback;
  const technical =
    /drift|evidently|feast|mlflow|production|reference|ks_|scipy|parquet/i.test(
      raw,
    );
  return technical ? fallback : raw;
}

export default function MonitoringPage() {
  const [status, setStatus] = useState<MonitoringStatus | null>(null);
  const [summary, setSummary] = useState<DriftSummary | null>(null);
  const [schedule, setSchedule] = useState<EvidentlySchedule | null>(null);
  const [dataVersions, setDataVersions] = useState<FeastVersionsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [reportHtml, setReportHtml] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);

  const [intervalHours, setIntervalHours] = useState(24);
  const [schedBuildRef, setSchedBuildRef] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [skipActivate, setSkipActivate] = useState(false);
  const [versionLabel, setVersionLabel] = useState("");

  const refresh = useCallback(async () => {
    setRefreshing(true);
    setError(null);
    try {
      const [s, sch, fv] = await Promise.all([
        getMonitoringStatus(),
        getEvidentlySchedule().catch(() => null),
        listFeastVersions().catch(() => null),
      ]);
      setStatus(s);
      setSummary(asSummary(s.last_drift_summary));
      if (sch) {
        setSchedule(sch);
        setIntervalHours(Number(sch.interval_hours) || 24);
        setSchedBuildRef(Boolean(sch.build_reference));
      }
      if (fv) setDataVersions(fv);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function onBuildReference() {
    setBusy("reference");
    setError(null);
    try {
      await buildReference();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function onRunDrift() {
    setBusy("drift");
    setError(null);
    try {
      const result = await runDrift({ minCurrentRows: 30 });
      setSummary(result);
      setShowReport(false);
      setReportHtml(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function onLoadReport() {
    setBusy("report");
    setReportError(null);
    try {
      const html = await fetchMonitoringReportHtml();
      setReportHtml(html);
      setShowReport(true);
    } catch (e) {
      setReportHtml(null);
      setShowReport(false);
      setReportError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function onSaveSchedule(enabled: boolean) {
    setBusy("schedule");
    setError(null);
    try {
      const sch = await setEvidentlySchedule({
        enabled,
        intervalHours,
        buildReference: schedBuildRef,
        minCurrentRows: 30,
      });
      setSchedule(sch);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function onClearSchedule() {
    setBusy("schedule");
    try {
      setSchedule(await clearEvidentlySchedule());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  async function onUploadData() {
    if (!uploadFile) {
      setError("Choose an Excel (.xlsx) or CSV file first.");
      return;
    }
    setBusy("upload");
    setError(null);
    try {
      await uploadFeastFile(uploadFile, {
        skipMaterialize: skipActivate,
        versionLabel: versionLabel.trim() || undefined,
      });
      setUploadFile(null);
      setVersionLabel("");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  const alertOn = Boolean(summary?.alert);
  const minRows = status?.min_rows_for_drift ?? 30;
  const rows = status?.production_log_rows ?? 0;
  const shiftPct =
    summary?.share_drifted != null
      ? `${(summary.share_drifted * 100).toFixed(0)}%`
      : "—";

  return (
    <div className="page">
      <section className="page-header">
        <div>
          <p className="eyebrow">Programme insights</p>
          <h2>Watch patterns &amp; refresh survey data</h2>
          <p className="lead">
            Compare recent screenings with your baseline, review quality reports, and
            keep teacher survey data up to date.
          </p>
        </div>
        <button
          type="button"
          className="ghost"
          onClick={() => void refresh()}
          disabled={Boolean(busy) || refreshing}
        >
          {refreshing ? (
            <>
              <span className="spinner" /> Updating…
            </>
          ) : (
            "Refresh"
          )}
        </button>
      </section>

      {alertOn && (
        <div className="card alert-banner" role="alert">
          <strong>Pattern change detected</strong>
          <p>{friendlyAlertMessage(summary?.alert_message)}</p>
        </div>
      )}

      <div className="cards-row">
        <div className="card metric">
          <span className="label">Recent screenings</span>
          <span className="value">{rows}</span>
          <span className="hint">
            {rows < minRows ? `${rows} of ${minRows} needed` : "Ready to compare"}
          </span>
        </div>
        <div className="card metric">
          <span className="label">Baseline</span>
          <span className="value">{status?.reference_exists ? "Ready" : "Not set"}</span>
          <span className="hint">Comparison starting point</span>
        </div>
        <div className="card metric">
          <span className="label">Changed share</span>
          <span className="value">{shiftPct}</span>
          <span className="hint">
            {summary?.dataset_drift == null
              ? "No check yet"
              : summary.dataset_drift
                ? "Shift flagged"
                : "Stable"}
          </span>
        </div>
        <div className="card metric">
          <span className="label">Last check</span>
          <span className="value small">
            {summary?.timestamp
              ? new Date(summary.timestamp).toLocaleString()
              : "None yet"}
          </span>
          <span className="hint">
            Baseline {summary?.n_reference ?? "—"} · Recent {summary?.n_current ?? "—"}
          </span>
        </div>
      </div>

      <div className="card panel actions">
        <div className="panel-head">
          <div>
            <h3>Quality checks</h3>
            <p className="hint">Compare recent screenings with the baseline and open the full report.</p>
          </div>
        </div>
        <div className="btn-row">
          <button
            type="button"
            onClick={() => void onBuildReference()}
            disabled={Boolean(busy)}
          >
            {busy === "reference" ? "Setting baseline…" : "Set baseline"}
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => void onRunDrift()}
            disabled={Boolean(busy) || rows < minRows}
          >
            {busy === "drift" ? "Checking…" : "Check for changes"}
          </button>
          <button
            type="button"
            onClick={() => void onLoadReport()}
            disabled={Boolean(busy)}
          >
            {busy === "report"
              ? "Opening…"
              : showReport
                ? "Reload report"
                : "Open report"}
          </button>
        </div>
        {rows < minRows && (
          <p className="hint soft-warn">
            {rows} of {minRows} recent screenings logged. Run more screenings before comparing.
          </p>
        )}
      </div>

      {error && (
        <div className="card error" role="alert">
          <strong>Something went wrong</strong>
          <p>{error}</p>
        </div>
      )}

      <div className="split-panels">
        <div className="card panel">
          <div className="panel-head">
            <div>
              <h3>Survey data updates</h3>
              <p className="hint">Upload a new survey file to refresh saved teacher profiles.</p>
            </div>
          </div>
          <div className="upload-stack">
            <label className="file-drop">
              <input
                type="file"
                accept=".xlsx,.xls,.csv"
                onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
              />
              <span className="file-drop-title">
                {uploadFile ? uploadFile.name : "Choose Excel or CSV"}
              </span>
              <span className="hint">Survey workbook or comma-separated file</span>
            </label>
            <div className="field">
              <label htmlFor="version_label">Version name (optional)</label>
              <input
                id="version_label"
                type="text"
                placeholder="e.g. Term 2 2026"
                value={versionLabel}
                onChange={(e) => setVersionLabel(e.target.value)}
              />
            </div>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={skipActivate}
                onChange={(e) => setSkipActivate(e.target.checked)}
              />
              Store only — do not activate for screening yet
            </label>
            <button
              type="button"
              className="primary"
              disabled={Boolean(busy) || !uploadFile}
              onClick={() => void onUploadData()}
            >
              {busy === "upload" ? "Uploading…" : "Upload & process"}
            </button>
          </div>

          {dataVersions?.current && (
            <p className="current-version">
              Active version <strong>{String(dataVersions.current.version)}</strong>
              {dataVersions.current.uploaded_at_utc
                ? ` · ${new Date(String(dataVersions.current.uploaded_at_utc)).toLocaleString()}`
                : ""}
            </p>
          )}

          {dataVersions && dataVersions.versions.length > 0 && (
            <div className="table-wrap">
              <table className="simple-table">
                <thead>
                  <tr>
                    <th>Version</th>
                    <th>File</th>
                    <th>Activated</th>
                  </tr>
                </thead>
                <tbody>
                  {dataVersions.versions.slice(0, 8).map((v) => (
                    <tr key={String(v.version)}>
                      <td>{String(v.version)}</td>
                      <td>{String(v.original_filename ?? "—")}</td>
                      <td>
                        {(v.pipeline as { materialized?: boolean } | undefined)
                          ?.materialized
                          ? "Yes"
                          : "No"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="card panel">
          <div className="panel-head">
            <div>
              <h3>Automatic checks</h3>
              <p className="hint">Run change detection on a recurring interval.</p>
            </div>
            <span className={`status-dot ${schedule?.enabled ? "on" : "off"}`}>
              {schedule?.enabled ? "On" : "Off"}
            </span>
          </div>
          <div className="schedule-grid">
            <div className="field">
              <label htmlFor="interval_hours">Every (hours)</label>
              <input
                id="interval_hours"
                type="number"
                min={0.25}
                step={0.25}
                value={intervalHours}
                onChange={(e) => setIntervalHours(Number(e.target.value) || 24)}
              />
            </div>
            <label className="checkbox">
              <input
                type="checkbox"
                checked={schedBuildRef}
                onChange={(e) => setSchedBuildRef(e.target.checked)}
              />
              Refresh baseline each run
            </label>
          </div>
          <div className="btn-row">
            <button
              type="button"
              className="primary"
              disabled={Boolean(busy)}
              onClick={() => void onSaveSchedule(true)}
            >
              {busy === "schedule" ? "Saving…" : "Turn on"}
            </button>
            <button
              type="button"
              className="ghost"
              disabled={Boolean(busy)}
              onClick={() => void onClearSchedule()}
            >
              Turn off
            </button>
          </div>
          {schedule && (
            <dl className="schedule-meta">
              {schedule.next_run_utc && (
                <>
                  <dt>Next</dt>
                  <dd>{new Date(schedule.next_run_utc).toLocaleString()}</dd>
                </>
              )}
              {schedule.last_run_utc && (
                <>
                  <dt>Last</dt>
                  <dd>
                    {new Date(schedule.last_run_utc).toLocaleString()}
                    {schedule.last_status ? ` · ${schedule.last_status}` : ""}
                  </dd>
                </>
              )}
              {schedule.last_error && (
                <>
                  <dt>Note</dt>
                  <dd className="warn-text">{schedule.last_error}</dd>
                </>
              )}
            </dl>
          )}
        </div>
      </div>

      {summary?.mean_shifts && Object.keys(summary.mean_shifts).length > 0 && (
        <div className="card panel">
          <div className="panel-head">
            <div>
              <h3>Score movement</h3>
              <p className="hint">Average changes between baseline and recent screenings.</p>
            </div>
          </div>
          <div className="table-wrap">
            <table className="simple-table">
              <thead>
                <tr>
                  <th>Measure</th>
                  <th>Baseline</th>
                  <th>Recent</th>
                  <th>Change</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(summary.mean_shifts)
                  .slice(0, 12)
                  .map(([name, v]) => (
                    <tr key={name}>
                      <td>{name.replace(/_/g, " ")}</td>
                      <td>{v.reference_mean?.toFixed?.(3) ?? "—"}</td>
                      <td>{v.current_mean?.toFixed?.(3) ?? "—"}</td>
                      <td>{v.delta?.toFixed?.(3) ?? "—"}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="card panel report">
        <div className="panel-head">
          <div>
            <h3>Detailed report</h3>
            <p className="hint">Full quality report for the latest change check.</p>
          </div>
        </div>
        {reportError && <p className="hint soft-warn">{reportError}</p>}
        {!showReport || !reportHtml ? (
          <div className="report-placeholder">
            <div>
              <p className="report-empty-title">No report open</p>
              <p className="hint">Open the latest quality report to review it here.</p>
              <div className="btn-row" style={{ justifyContent: "center" }}>
                <button
                  type="button"
                  className="primary"
                  onClick={() => void onLoadReport()}
                  disabled={Boolean(busy)}
                >
                  Open report
                </button>
              </div>
            </div>
          </div>
        ) : (
          <iframe
            title="Quality report"
            className="report-frame"
            srcDoc={reportHtml}
            sandbox="allow-same-origin allow-scripts allow-popups"
          />
        )}
      </div>
    </div>
  );
}
