import type {
  ApiErrorBody,
  DriftSummary,
  EvidentlySchedule,
  FeastVersionsResponse,
  HealthResponse,
  MonitoringStatus,
  PredictPayload,
  PredictionResult,
} from "../types";

export const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000";

async function parseError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as ApiErrorBody;
    const d = body.detail;
    if (typeof d === "string") return d;
    if (d && typeof d === "object") {
      const parts = [d.message, ...(d.errors || [])].filter(Boolean);
      return parts.join(" — ") || res.statusText;
    }
  } catch {
    /* ignore */
  }
  return `${res.status} ${res.statusText}`;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health?t=${Date.now()}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function predict(
  payload: PredictPayload,
): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function listFeastTeachers(limit = 20): Promise<{
  teacher_ids?: number[];
  teachers?: Array<{ teacher_id: number }>;
  count?: number;
}> {
  const res = await fetch(`${API_BASE}/feast/teachers?limit=${limit}`);
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function getMonitoringStatus(): Promise<MonitoringStatus> {
  const res = await fetch(`${API_BASE}/monitoring/status?t=${Date.now()}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function buildReference(maxRows = 5000): Promise<unknown> {
  const res = await fetch(
    `${API_BASE}/monitoring/build-reference?max_rows=${maxRows}`,
    { method: "POST" },
  );
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function runDrift(opts?: {
  buildReference?: boolean;
  minCurrentRows?: number;
}): Promise<DriftSummary> {
  const params = new URLSearchParams();
  if (opts?.buildReference) params.set("build_reference", "true");
  if (opts?.minCurrentRows != null)
    params.set("min_current_rows", String(opts.minCurrentRows));
  const q = params.toString();
  const res = await fetch(
    `${API_BASE}/monitoring/run-drift${q ? `?${q}` : ""}`,
    { method: "POST" },
  );
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

/** Fetch HTML as text for iframe srcdoc (avoids download / X-Frame issues). */
export async function fetchMonitoringReportHtml(): Promise<string> {
  const res = await fetch(
    `${API_BASE}/monitoring/report?t=${Date.now()}`,
    { cache: "no-store", headers: { Accept: "text/html" } },
  );
  if (!res.ok) throw new Error(await parseError(res));
  return res.text();
}

export function monitoringReportUrl(): string {
  return `${API_BASE}/monitoring/report?inline=1&t=${Date.now()}`;
}

export async function getEvidentlySchedule(): Promise<EvidentlySchedule> {
  const res = await fetch(`${API_BASE}/monitoring/schedule?t=${Date.now()}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function setEvidentlySchedule(opts: {
  enabled: boolean;
  intervalHours: number;
  buildReference?: boolean;
  minCurrentRows?: number;
}): Promise<EvidentlySchedule> {
  const params = new URLSearchParams({
    enabled: String(opts.enabled),
    interval_hours: String(opts.intervalHours),
    build_reference: String(Boolean(opts.buildReference)),
    min_current_rows: String(opts.minCurrentRows ?? 30),
  });
  const res = await fetch(`${API_BASE}/monitoring/schedule?${params}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function clearEvidentlySchedule(): Promise<EvidentlySchedule> {
  const res = await fetch(`${API_BASE}/monitoring/schedule`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function listFeastVersions(): Promise<FeastVersionsResponse> {
  const res = await fetch(`${API_BASE}/feast/versions?t=${Date.now()}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function uploadFeastFile(
  file: File,
  opts?: { skipMaterialize?: boolean; versionLabel?: string },
): Promise<Record<string, unknown>> {
  const form = new FormData();
  form.append("file", file);
  form.append("skip_materialize", String(Boolean(opts?.skipMaterialize)));
  if (opts?.versionLabel) form.append("version_label", opts.versionLabel);
  const res = await fetch(`${API_BASE}/feast/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}
