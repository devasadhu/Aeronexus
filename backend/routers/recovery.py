"""
backend/routers/recovery.py

Dedicated recovery router — cleaner than the disruptions router's /recover endpoint.
Adds audit trail recording on every plan generation.

Endpoints:
  POST /recovery/run          Run full pipeline for a flight_id
  GET  /recovery/audit        All audit records (newest first)
  GET  /recovery/audit/{fid}  History for one flight
  GET  /recovery/audit/diff   Diff two versions
  GET  /recovery/health       Network health score
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import json, pickle
from pathlib import Path
from datetime import datetime

router = APIRouter()


def _load_json(path: str):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []


def _load_graph():
    p = Path("data/processed/flight_graph.gpickle")
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


# ── POST /recovery/run ────────────────────────────────────────────────────────

@router.post("/run")
def run_recovery(
    flight_id:      str   = Query(..., description="Root disrupted flight ID"),
    risk_threshold: float = Query(0.25, ge=0.0, le=1.0),
    scenario_name:  Optional[str] = Query(None),
    trigger:        str   = Query("api"),
):
    """
    Full pipeline: cascade → severity → recovery plan → audit record.
    Returns plan + audit record + network health.
    """
    from ml.feature_builder  import build_historical_rates, build_downstream_index
    from ml.cascade_model    import predict_cascade
    from ml.severity_scorer  import build_disruption_event
    from agents.coordinator  import run_recovery_pipeline
    from backend.audit_trail import record_plan, get_network_health

    disruptions = _load_json("data/processed/disruptions_seed.json")
    flights     = _load_json("data/processed/flights.json")
    G           = _load_graph()

    disruption = next((d for d in disruptions if d["flight_id"] == flight_id), None)
    if not disruption:
        raise HTTPException(404, detail=f"No disruption found for flight_id={flight_id}")

    hist_rates     = build_historical_rates(flights)
    downstream_idx = build_downstream_index(flights)
    flights_lookup = {f["id"]: f for f in flights}

    candidates = [
        f for f in flights
        if f["origin_id"] == disruption.get("destination", "")
        and f["status"] not in ("cancelled",)
    ][:50]

    affected = predict_cascade(
        disruption, candidates, G,
        hist_rates, downstream_idx,
        risk_threshold=risk_threshold,
    )
    event = build_disruption_event(disruption, affected, flights_lookup, G)

    crew        = _load_json("data/processed/crew_roster.json")
    fleet       = _load_json("data/processed/aircraft_fleet.json")
    passengers  = _load_json("data/processed/passengers.json")
    itineraries = _load_json("data/processed/itineraries.json")

    affected_ids  = {a["flight_id"] for a in affected}
    affected_full = [f for f in flights if f["id"] in affected_ids]

    plan = run_recovery_pipeline(
        disruption         = event,
        affected_flights   = affected_full,
        all_flights        = flights,
        available_crew     = crew,
        available_aircraft = fleet,
        passengers         = passengers,
        itineraries        = itineraries,
    )

    # record in audit trail
    audit_record = record_plan(
        disruption_id = disruption.get("id", flight_id),
        flight_id     = flight_id,
        plan          = plan,
        scenario_name = scenario_name,
        trigger       = trigger,
    )

    # network health snapshot
    health = get_network_health(flights)

    return {
        "disruption_event": event,
        "recovery_plan":    plan,
        "audit_record":     audit_record,
        "network_health":   health,
    }


# ── GET /recovery/audit ───────────────────────────────────────────────────────

@router.get("/audit")
def list_audit(limit: int = Query(50, le=200)):
    from backend.audit_trail import get_all_history
    return get_all_history(limit=limit)


@router.get("/audit/{flight_id}")
def flight_audit(flight_id: str):
    from backend.audit_trail import get_history
    history = get_history(flight_id)
    if not history:
        raise HTTPException(404, detail=f"No audit records for flight {flight_id}")
    return history


@router.get("/audit/{flight_id}/diff")
def version_diff(
    flight_id: str,
    v1: int = Query(...),
    v2: int = Query(...),
):
    from backend.audit_trail import diff_versions
    return diff_versions(flight_id, v1, v2)


# ── GET /recovery/health ──────────────────────────────────────────────────────

@router.get("/health")
def network_health():
    from backend.audit_trail import get_network_health
    flights = _load_json("data/processed/flights.json")
    return get_network_health(flights)