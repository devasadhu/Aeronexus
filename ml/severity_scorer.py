"""
ml/severity_scorer.py

Computes a structured DisruptionEvent from:
  - Root disruption metadata
  - Cascade model predictions (affected flights + risk scores)
  - Network graph (hub status, connectivity)

Outputs a DisruptionEvent-compatible dict ready for:
  - DB insertion
  - Agent pipeline input
  - Advisory generation
"""

import sys, json, logging
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

HUB_AIRPORTS = {
    "ATL","ORD","LAX","DFW","DEN","JFK","SFO","LHR","CDG","AMS",
    "FRA","DXB","SIN","DEL","BOM","NRT","ICN","EWR","IAH","BOS",
}

# Severity thresholds
SEVERITY_RULES = [
    # (condition_fn, severity)
    (lambda s: s["score"] >= 80,  "critical"),
    (lambda s: s["score"] >= 50,  "high"),
    (lambda s: s["score"] >= 25,  "medium"),
    (lambda s: True,               "low"),
]


def compute_severity_score(
    disruption: dict,
    affected_flights: list,
    passengers_affected: int,
    is_hub_origin: bool,
    time_of_day: int,
) -> dict:
    """
    Weighted severity scoring.
    Returns {"score": 0-100, "breakdown": {...}}
    """
    breakdown = {}

    # 1. Delay magnitude (0–25 pts)
    delay = disruption.get("delay_minutes", 0)
    delay_pts = min(25, delay / 8)
    breakdown["delay_magnitude"] = round(delay_pts, 1)

    # 2. Number of affected flights (0–25 pts)
    # Each affected flight with risk > 0.5 = full weight; lower risk = partial
    high_risk = sum(1 for f in affected_flights if f.get("risk_score", 0) >= 0.5)
    med_risk  = sum(1 for f in affected_flights if 0.3 <= f.get("risk_score", 0) < 0.5)
    cascade_pts = min(25, high_risk * 3 + med_risk * 1.5)
    breakdown["cascade_breadth"] = round(cascade_pts, 1)

    # 3. Passenger impact (0–20 pts)
    pax_pts = min(20, passengers_affected / 25)
    breakdown["passenger_impact"] = round(pax_pts, 1)

    # 4. Hub penalty (0–15 pts) — disruptions at hubs propagate further
    hub_pts = 15 if is_hub_origin else 5
    breakdown["hub_penalty"] = hub_pts

    # 5. Time-of-day amplifier (0–15 pts)
    # Peak hours (7–9, 16–20) = more connections at risk
    if 7 <= time_of_day <= 9 or 16 <= time_of_day <= 20:
        tod_pts = 15
    elif 22 <= time_of_day or time_of_day <= 5:
        tod_pts = 5   # off-peak, fewer connections
    else:
        tod_pts = 10

    breakdown["time_of_day"] = tod_pts

    # 6. Disruption type multiplier
    type_mult = {
        "weather": 1.3,
        "mechanical": 1.1,
        "carrier": 1.0,
        "atc": 1.2,
        "airport": 1.4,
        "unknown": 1.0,
    }.get(disruption.get("type", "unknown"), 1.0)

    raw_score = sum(breakdown.values())
    score = min(100, raw_score * type_mult)
    breakdown["type_multiplier"] = type_mult

    return {"score": round(score, 1), "breakdown": breakdown}


def score_to_severity(score_dict: dict) -> str:
    for condition, severity in SEVERITY_RULES:
        if condition(score_dict):
            return severity
    return "low"


def build_disruption_event(
    disruption: dict,
    affected_flights: list,
    flights_lookup: dict,
    G=None,
    avg_pax_per_flight: int = 140,
) -> dict:
    """
    Builds a complete DisruptionEvent dict.

    disruption     = seed dict from disruptions_seed.json
    affected_flights = output of predict_cascade()
    flights_lookup   = {flight_id: flight_dict}
    G              = NetworkX graph (optional, for hub check)
    """
    root_flight = flights_lookup.get(disruption["flight_id"], {})
    origin      = disruption.get("origin", root_flight.get("origin_id", ""))
    is_hub      = origin in HUB_AIRPORTS

    dep_time_str = disruption.get("departure_time", datetime.utcnow().isoformat())
    try:
        dep_time = datetime.fromisoformat(dep_time_str)
    except ValueError:
        dep_time = datetime.utcnow()

    passengers_affected = len(affected_flights) * avg_pax_per_flight

    score_dict = compute_severity_score(
        disruption         = disruption,
        affected_flights   = affected_flights,
        passengers_affected = passengers_affected,
        is_hub_origin      = is_hub,
        time_of_day        = dep_time.hour,
    )
    severity = score_to_severity(score_dict)

    total_delay = sum(f.get("delay_estimate_min", 0) for f in affected_flights)
    avg_delay   = int(total_delay / len(affected_flights)) if affected_flights else 0

    event = {
        "id":                     None,   # filled by DB on insert
        "root_flight_id":         disruption["flight_id"],
        "detected_at":            datetime.utcnow().isoformat(),
        "disruption_type":        disruption.get("type", "unknown"),
        "severity":               severity,
        "severity_score":         score_dict["score"],
        "severity_breakdown":     score_dict["breakdown"],
        "affected_flights_json":  affected_flights,
        "total_affected_flights": len(affected_flights),
        "total_affected_pax":     passengers_affected,
        "estimated_delay_min":    avg_delay,
        "weather_metar":          disruption.get("weather_metar"),
        "weather_condition":      _weather_label(disruption),
        "resolved":               False,
        "resolved_at":            None,
        "recovery_plan_id":       None,
    }

    return event


def _weather_label(disruption: dict) -> Optional[str]:
    dtype = disruption.get("type", "")
    delay = disruption.get("delay_minutes", 0)
    if dtype == "weather":
        if delay > 120:  return "LIFR"
        elif delay > 60: return "IFR"
        else:            return "MVFR"
    return None


def summarise_event(event: dict) -> str:
    sev = event["severity"].upper()
    n   = event["total_affected_flights"]
    pax = event["total_affected_pax"]
    delay = event["estimated_delay_min"]
    dtype = event["disruption_type"]
    return (
        f"[{sev}] {dtype.upper()} disruption | "
        f"{n} downstream flights at risk | "
        f"~{pax} passengers | "
        f"avg est. delay {delay} min"
    )


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle

    flights = json.loads(Path("data/processed/flights.json").read_text())
    disruptions = json.loads(Path("data/processed/disruptions_seed.json").read_text())
    flights_lookup = {f["id"]: f for f in flights}

    G = None
    graph_path = Path("data/processed/flight_graph.gpickle")
    if graph_path.exists():
        with open(graph_path, "rb") as f:
            G = pickle.load(f)

    from ml.feature_builder import build_historical_rates, build_downstream_index
    from ml.cascade_model import predict_cascade

    hist_rates     = build_historical_rates(flights)
    downstream_idx = build_downstream_index(flights)

    for disruption in disruptions[:3]:
        candidates = [f for f in flights
                       if f["origin_id"] == disruption.get("destination", "")
                       and f["status"] not in ("cancelled",)][:20]

        affected = predict_cascade(disruption, candidates, G, hist_rates,
                                    downstream_idx, risk_threshold=0.25)

        event = build_disruption_event(disruption, affected, flights_lookup, G)
        log.info(f"\nFlight {disruption['flight_number']}: {summarise_event(event)}")
        log.info(f"  Score breakdown: {event['severity_breakdown']}")