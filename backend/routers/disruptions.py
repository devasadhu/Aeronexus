from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import json, pickle
from pathlib import Path
from datetime import datetime

router = APIRouter()

_disruptions_cache = None
_flights_cache     = None
_graph_cache       = None


def _load_disruptions():
    global _disruptions_cache
    if _disruptions_cache is None:
        p = Path("data/processed/disruptions_seed.json")
        _disruptions_cache = json.loads(p.read_text()) if p.exists() else []
    return _disruptions_cache


def _load_flights():
    global _flights_cache
    if _flights_cache is None:
        p = Path("data/processed/flights.json")
        _flights_cache = json.loads(p.read_text()) if p.exists() else []
    return _flights_cache


def _load_graph():
    global _graph_cache
    if _graph_cache is None:
        p = Path("data/processed/flight_graph.gpickle")
        if p.exists():
            with open(p, "rb") as f:
                _graph_cache = pickle.load(f)
    return _graph_cache


@router.get("/")
def list_disruptions(
    severity: Optional[str] = Query(None),
    limit:    int           = Query(20, le=100),
    offset:   int           = Query(0),
):
    disruptions = _load_disruptions()
    if severity:
        # severity not stored in seed; return all for now
        pass
    return disruptions[offset: offset + limit]


@router.get("/{disruption_id}")
def get_disruption(disruption_id: str):
    disruptions = _load_disruptions()
    for d in disruptions:
        if d["flight_id"] == disruption_id or disruptions.index(d) == int(disruption_id or -1):
            return d
    raise HTTPException(status_code=404, detail="Disruption not found")


@router.get("/{flight_id}/cascade")
def get_cascade(
    flight_id:      str,
    risk_threshold: float = Query(0.25, ge=0.0, le=1.0),
    max_results:    int   = Query(20, le=50),
):
    """
    Run cascade prediction for a disruption rooted at flight_id.
    """
    disruptions = _load_disruptions()
    flights     = _load_flights()
    G           = _load_graph()

    disruption = next((d for d in disruptions if d["flight_id"] == flight_id), None)
    if not disruption:
        raise HTTPException(status_code=404, detail="No disruption found for this flight_id")

    from ml.feature_builder import build_historical_rates, build_downstream_index
    from ml.cascade_model   import predict_cascade

    hist_rates     = build_historical_rates(flights)
    downstream_idx = build_downstream_index(flights)

    candidates = [
        f for f in flights
        if f["origin_id"] == disruption.get("destination", "")
        and f["status"] not in ("cancelled",)
    ][:50]

    affected = predict_cascade(
        disruption, candidates, G,
        hist_rates, downstream_idx,
        risk_threshold=risk_threshold,
        max_results=max_results,
    )

    from ml.severity_scorer import build_disruption_event, summarise_event
    flights_lookup = {f["id"]: f for f in flights}
    event = build_disruption_event(disruption, affected, flights_lookup, G)

    return {
        "disruption":      disruption,
        "event":           event,
        "summary":         summarise_event(event),
        "affected_count":  len(affected),
        "affected_flights": affected,
    }


@router.post("/{flight_id}/recover")
def trigger_recovery(
    flight_id:      str,
    risk_threshold: float = Query(0.25),
):
    """
    Run full recovery pipeline for a disruption.
    Returns complete RecoveryPlan.
    """
    disruptions = _load_disruptions()
    flights     = _load_flights()
    G           = _load_graph()

    disruption = next((d for d in disruptions if d["flight_id"] == flight_id), None)
    if not disruption:
        raise HTTPException(status_code=404, detail="No disruption found for this flight_id")

    from ml.feature_builder  import build_historical_rates, build_downstream_index
    from ml.cascade_model    import predict_cascade
    from ml.severity_scorer  import build_disruption_event

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

    # load synthetic operational data
    crew_path  = Path("data/processed/crew_roster.json")
    fleet_path = Path("data/processed/aircraft_fleet.json")
    pax_path   = Path("data/processed/passengers.json")
    itin_path  = Path("data/processed/itineraries.json")

    crew       = json.loads(crew_path.read_text())  if crew_path.exists()  else []
    fleet      = json.loads(fleet_path.read_text()) if fleet_path.exists() else []
    passengers = json.loads(pax_path.read_text())   if pax_path.exists()   else []
    itineraries = json.loads(itin_path.read_text()) if itin_path.exists()  else []

    # get full flight dicts for affected
    affected_flight_ids = {a["flight_id"] for a in affected}
    affected_flights_full = [f for f in flights if f["id"] in affected_flight_ids]

    from agents.coordinator import run_recovery_pipeline
    plan = run_recovery_pipeline(
        disruption         = event,
        affected_flights   = affected_flights_full,
        all_flights        = flights,
        available_crew     = crew,
        available_aircraft = fleet,
        passengers         = passengers,
        itineraries        = itineraries,
    )

    return {
        "disruption_event": event,
        "recovery_plan":    plan,
    }