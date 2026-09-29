"""
In-process scheduler for Evidently drift checks (platform-triggered).

Stores config under data/monitoring/evidently_schedule.json.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEDULE_PATH = PROJECT_ROOT / "data" / "monitoring" / "evidently_schedule.json"

_lock = threading.Lock()
_timer: Optional[threading.Timer] = None
_run_fn: Optional[Callable[[], Dict[str, Any]]] = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _default_schedule() -> Dict[str, Any]:
    return {
        "enabled": False,
        "interval_hours": 24,
        "build_reference": False,
        "min_current_rows": 30,
        "next_run_utc": None,
        "last_run_utc": None,
        "last_status": None,
        "last_error": None,
        "last_summary": None,
    }


def load_schedule() -> Dict[str, Any]:
    if not SCHEDULE_PATH.exists():
        return _default_schedule()
    try:
        data = json.loads(SCHEDULE_PATH.read_text(encoding="utf-8"))
        base = _default_schedule()
        base.update(data or {})
        return base
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read schedule file: %s", exc)
        return _default_schedule()


def save_schedule(data: Dict[str, Any]) -> Dict[str, Any]:
    SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def configure_runner(fn: Callable[[], Dict[str, Any]]) -> None:
    """Register the callable that executes one drift check."""
    global _run_fn
    _run_fn = fn


def _cancel_timer() -> None:
    global _timer
    if _timer is not None:
        _timer.cancel()
        _timer = None


def _arm_timer(seconds: float) -> None:
    global _timer
    _cancel_timer()
    if seconds <= 0:
        seconds = 60.0
    _timer = threading.Timer(seconds, _on_timer)
    _timer.daemon = True
    _timer.start()


def _on_timer() -> None:
    with _lock:
        sched = load_schedule()
        if not sched.get("enabled"):
            return
        try:
            if _run_fn is None:
                raise RuntimeError("No drift runner configured")
            summary = _run_fn()
            sched["last_run_utc"] = _now().isoformat()
            sched["last_status"] = "ok"
            sched["last_error"] = None
            sched["last_summary"] = {
                k: summary.get(k)
                for k in (
                    "alert",
                    "share_drifted",
                    "dataset_drift",
                    "n_current",
                    "n_reference",
                    "timestamp",
                )
                if isinstance(summary, dict)
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Scheduled Evidently job failed")
            sched["last_run_utc"] = _now().isoformat()
            sched["last_status"] = "error"
            sched["last_error"] = str(exc)
        hours = float(sched.get("interval_hours") or 24)
        nxt = _now() + timedelta(hours=hours)
        sched["next_run_utc"] = nxt.isoformat()
        save_schedule(sched)
        if sched.get("enabled"):
            _arm_timer(hours * 3600.0)


def start_or_refresh_scheduler() -> Dict[str, Any]:
    """Start/restart timer from saved schedule (call on API startup)."""
    with _lock:
        sched = load_schedule()
        _cancel_timer()
        if not sched.get("enabled"):
            return sched
        hours = float(sched.get("interval_hours") or 24)
        next_s = sched.get("next_run_utc")
        if next_s:
            try:
                nxt = datetime.fromisoformat(str(next_s).replace("Z", "+00:00"))
                delay = (nxt - _now()).total_seconds()
            except Exception:  # noqa: BLE001
                delay = hours * 3600.0
        else:
            delay = hours * 3600.0
            sched["next_run_utc"] = (_now() + timedelta(hours=hours)).isoformat()
            save_schedule(sched)
        _arm_timer(max(30.0, delay))
        return sched


def stop_scheduler() -> Dict[str, Any]:
    with _lock:
        sched = load_schedule()
        sched["enabled"] = False
        sched["next_run_utc"] = None
        save_schedule(sched)
        _cancel_timer()
        return sched


def upsert_schedule(
    *,
    enabled: bool,
    interval_hours: float = 24.0,
    build_reference: bool = False,
    min_current_rows: int = 30,
) -> Dict[str, Any]:
    with _lock:
        sched = load_schedule()
        sched["enabled"] = bool(enabled)
        sched["interval_hours"] = float(max(0.25, interval_hours))  # min 15 minutes
        sched["build_reference"] = bool(build_reference)
        sched["min_current_rows"] = int(max(1, min_current_rows))
        if enabled:
            sched["next_run_utc"] = (
                _now() + timedelta(hours=sched["interval_hours"])
            ).isoformat()
        else:
            sched["next_run_utc"] = None
        save_schedule(sched)
    return start_or_refresh_scheduler()
