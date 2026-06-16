"""
backend/audit_trail.py

Recovery plan audit trail — stores every plan generated for a disruption,
diffs successive versions, and exposes a history log.

Storage: JSON file at data/processed/audit_trail.json (no DB required).
Each entry is a PlanRecord with a version number, timestamp, summary diff,
and the full plan snapshot.

Used by:
  - routers/recovery.py  (write on each /recover call)
  - routers/audit.py     (read history, diff two versions)
  - dashboard OCC Assistant (context for chatbot)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

AUDIT_PATH = Path("data/processed/audit_trail.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_trail() -> list[dict]:
    if AUDIT_PATH.exists():
        try:
            return json.loads(AUDIT_PATH.read_text())
        except Exception:
            pass
    return []


def _save_trail(trail: list[dict]) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(trail, indent=2, default=str))


def _summarise(plan: dict) -> dict:
    """Compact summary of a plan for diff display."""
    summary = plan.get("summary", {})
    actions = plan.get("final_actions_json", [])
    return {
        "total_actions":         len(actions),
        "feasible_actions":      sum(1 for a in actions if a.get("feasible")),
        "cancellations_avoided": summary.get("cancellations_avoided", 0),
        "misconnects_avoided":   summary.get("misconnects_avoided", 0),
        "delay_reduction_min":   summary.get("total_delay_reduction_min", 0),
        "conflict_count":        summary.get("conflict_count", 0),
        "flights_cancelled":     summary.get("flights_cancelled", 0),
        "rebooked_pax":          sum(
            1 for a in actions
            if a.get("action_type") == "passenger_rebook" and a.get("feasible")
        ),
        "stranded_pax":          sum(
            1 for a in actions
            if a.get("action_type") == "passenger_rebook" and not a.get("feasible")
        ),
    }


def _diff_summaries(prev: dict, curr: dict) -> dict:
    """Returns field-level delta between two plan summaries."""
    delta = {}
    for k in curr:
        old = prev.get(k, 0)
        new = curr.get(k, 0)
        if old != new:
            delta[k] = {"from": old, "to": new, "delta": new - old}
    return delta


# ── public API ────────────────────────────────────────────────────────────────

def record_plan(
    disruption_id: str,
    flight_id:     str,
    plan:          dict,
    scenario_name: Optional[str] = None,
    trigger:       str = "api",
) -> dict:
    """
    Save a new plan version for a disruption.
    Returns the audit record created.
    """
    trail = _load_trail()

    # find previous version for this disruption
    prev_versions = [
        r for r in trail
        if r["flight_id"] == flight_id
    ]
    version = len(prev_versions) + 1
    prev_summary = prev_versions[-1]["plan_summary"] if prev_versions else {}

    curr_summary = _summarise(plan)
    diff         = _diff_summaries(prev_summary, curr_summary) if prev_summary else {}

    record = {
        "id":            f"{flight_id}_v{version}",
        "flight_id":     flight_id,
        "disruption_id": disruption_id,
        "version":       version,
        "timestamp":     _now_iso(),
        "trigger":       trigger,        # "api" | "dashboard" | "scenario" | "auto"
        "scenario_name": scenario_name,
        "plan_summary":  curr_summary,
        "diff_from_prev": diff,
        "plan_snapshot": {               # store lightweight snapshot, not full plan
            "advisory_text":   plan.get("advisory_text"),
            "summary":         plan.get("summary", {}),
            "cancelled_flights": plan.get("cancelled_flights", []),
            "conflict_count":  plan.get("conflict_count", 0),
        },
    }

    trail.append(record)
    _save_trail(trail)

    log.info(
        f"[Audit] recorded plan v{version} for flight {flight_id} "
        f"(trigger={trigger}, diff_fields={list(diff.keys())})"
    )
    return record


def get_history(flight_id: str) -> list[dict]:
    """All audit records for a flight, newest first."""
    trail = _load_trail()
    recs  = [r for r in trail if r["flight_id"] == flight_id]
    return sorted(recs, key=lambda r: r["timestamp"], reverse=True)


def get_all_history(limit: int = 50) -> list[dict]:
    """All audit records across all flights, newest first."""
    trail = _load_trail()
    return sorted(trail, key=lambda r: r["timestamp"], reverse=True)[:limit]


def diff_versions(flight_id: str, v1: int, v2: int) -> dict:
    """
    Compare two specific versions of a plan for a given flight.
    Returns summary diff + metadata for both versions.
    """
    trail = _load_trail()
    recs  = {
        r["version"]: r
        for r in trail if r["flight_id"] == flight_id
    }
    rec1 = recs.get(v1)
    rec2 = recs.get(v2)
    if not rec1 or not rec2:
        return {"error": f"Version(s) not found for flight {flight_id}"}

    return {
        "flight_id": flight_id,
        "v1":        {"version": v1, "timestamp": rec1["timestamp"], "summary": rec1["plan_summary"]},
        "v2":        {"version": v2, "timestamp": rec2["timestamp"], "summary": rec2["plan_summary"]},
        "diff":      _diff_summaries(rec1["plan_summary"], rec2["plan_summary"]),
    }


def get_network_health(flights: list) -> dict:
    """
    Compute a live network health score from current flight data.
    Score: 0 (all cancelled/delayed) → 100 (all on time).
    Called by the dashboard heartbeat.
    """
    if not flights:
        return {"score": 100, "details": {}}

    total     = len(flights)
    cancelled = sum(1 for f in flights if f.get("status") == "cancelled")
    delayed   = sum(1 for f in flights if f.get("delay_minutes", 0) > 15)
    on_time   = total - cancelled - delayed

    avg_delay = (
        sum(f.get("delay_minutes", 0) for f in flights) / total
    )

    # weighted score
    score = max(0, min(100, round(
        (on_time / total) * 70 +
        max(0, 1 - avg_delay / 120) * 20 +
        max(0, 1 - cancelled / max(total, 1)) * 10
    )))

    return {
        "score":       score,
        "total":       total,
        "on_time":     on_time,
        "delayed":     delayed,
        "cancelled":   cancelled,
        "avg_delay_min": round(avg_delay, 1),
        "computed_at": _now_iso(),
    }