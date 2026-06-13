from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Optional
import os, json

router = APIRouter()

ADVISORY_SYSTEM_PROMPT = """You are an Airline Operations Control Centre (OCC) AI assistant.
Given a flight disruption event and its recovery plan, produce a concise operational advisory.

Format:
1. One-sentence situation summary (flight, cause, severity)
2. Bullet list of recovery actions taken (fleet, crew, passengers)
3. One-sentence risk note (what remains unresolved)

Use airline OCC language. Be direct. Max 200 words total."""


def _format_plan_for_llm(event: dict, plan: dict) -> str:
    summary = plan.get("summary", {})
    actions = plan.get("final_actions_json", [])

    swaps    = [a for a in actions if a["action_type"] == "aircraft_swap"   and a["feasible"]]
    crew_r   = [a for a in actions if a["action_type"] == "crew_reassign"   and a["feasible"]]
    rebooks  = [a for a in actions if a["action_type"] == "passenger_rebook" and a["feasible"]]
    cancels  = [a for a in actions if a["action_type"] == "flight_cancel"]
    stranded = [a for a in actions if a["action_type"] == "passenger_rebook" and not a["feasible"]]

    lines = [
        f"DISRUPTION: {event.get('disruption_type','unknown').upper()} | "
        f"Severity: {event.get('severity','?').upper()} | "
        f"Score: {event.get('severity_score', '?')}",
        f"Root flight: {event.get('root_flight_id','?')}",
        f"Affected flights: {event.get('total_affected_flights', 0)}",
        f"Affected passengers: {event.get('total_affected_pax', 0)}",
        f"Estimated avg delay: {event.get('estimated_delay_min', 0)} min",
        "",
        "RECOVERY ACTIONS:",
        f"  Aircraft swaps proposed:   {len(swaps)}",
        f"  Crew reassignments:        {len(crew_r)}",
        f"  Passengers rebooked:       {len(rebooks)}",
        f"  Flights cancelled:         {len(cancels)}",
        f"  Stranded passengers:       {len(stranded)}",
        f"  Conflicts detected:        {summary.get('conflict_count', 0)}",
        "",
        "METRICS:",
        f"  Cancellations avoided:     {summary.get('cancellations_avoided', 0)}",
        f"  Misconnects avoided:       {summary.get('misconnects_avoided', 0)}",
        f"  Total delay reduction:     {summary.get('total_delay_reduction_min', 0)} min",
    ]

    if swaps:
        lines.append(f"\nFLEET: {swaps[0]['description']}" +
                     (f" (+{len(swaps)-1} more)" if len(swaps) > 1 else ""))
    if crew_r:
        lines.append(f"CREW: {crew_r[0]['description']}" +
                     (f" (+{len(crew_r)-1} more)" if len(crew_r) > 1 else ""))
    if rebooks:
        lines.append(f"PAX: {rebooks[0]['description']}" +
                     (f" (+{len(rebooks)-1} more)" if len(rebooks) > 1 else ""))

    return "\n".join(lines)


def _call_groq(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        # Fallback: generate a rule-based advisory without LLM
        return None

    try:
        import groq
        client = groq.Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": ADVISORY_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return None


def _rule_based_advisory(event: dict, plan: dict) -> tuple:
    """Fallback advisory when Groq is not configured."""
    summary = plan.get("summary", {})
    actions = plan.get("final_actions_json", [])

    sev    = event.get("severity", "unknown").upper()
    dtype  = event.get("disruption_type", "unknown")
    n_aff  = event.get("total_affected_flights", 0)
    n_pax  = event.get("total_affected_pax", 0)
    delay  = event.get("estimated_delay_min", 0)

    swaps   = sum(1 for a in actions if a["action_type"] == "aircraft_swap"   and a["feasible"])
    crew_r  = sum(1 for a in actions if a["action_type"] == "crew_reassign"   and a["feasible"])
    rebooks = sum(1 for a in actions if a["action_type"] == "passenger_rebook" and a["feasible"])
    cancels = sum(1 for a in actions if a["action_type"] == "flight_cancel")
    stranded = sum(1 for a in actions if a["action_type"] == "passenger_rebook" and not a["feasible"])

    text = (
        f"[{sev}] {dtype.upper()} disruption affecting {n_aff} flights "
        f"and approximately {n_pax} passengers with estimated average delay of {delay} min. "
        f"Recovery plan: {swaps} aircraft swap(s), {crew_r} crew reassignment(s), "
        f"{rebooks} passenger rebooking(s). "
        f"{cancels} flight(s) cancelled. "
        f"{'WARNING: ' + str(stranded) + ' passenger(s) require manual rebooking. ' if stranded else ''}"
        f"Plan generated at {datetime.utcnow().strftime('%H:%MZ')}."
    )

    bullets = []
    if swaps:
        bullets.append(f"Fleet: {swaps} aircraft swap(s) arranged to maintain rotations")
    if crew_r:
        bullets.append(f"Crew: {crew_r} legal reassignment(s) from available/standby pool")
    if rebooks:
        bullets.append(f"Passengers: {rebooks} booking(s) rerouted to alternative flights")
    if cancels:
        bullets.append(f"Operations: {cancels} flight(s) cancelled due to resource unavailability")
    if stranded:
        bullets.append(f"Action required: {stranded} passenger(s) need manual rebooking")

    return text, bullets


@router.post("/generate")
def generate_advisory(body: dict):
    """
    Body: { "disruption_event": {...}, "recovery_plan": {...} }
    Returns: { "summary": str, "bullets": [...], "generated_at": ISO }
    """
    event = body.get("disruption_event")
    plan  = body.get("recovery_plan")

    if not event or not plan:
        raise HTTPException(status_code=400,
                             detail="Both disruption_event and recovery_plan required")

    prompt = _format_plan_for_llm(event, plan)
    llm_text = _call_groq(prompt)

    if llm_text:
        # parse LLM output into summary + bullets
        lines   = [l.strip() for l in llm_text.split("\n") if l.strip()]
        summary = lines[0] if lines else llm_text
        bullets = [l.lstrip("•-* ") for l in lines[1:] if l.startswith(("•", "-", "*", " "))]
        if not bullets:
            bullets = lines[1:]
    else:
        summary, bullets = _rule_based_advisory(event, plan)

    return {
        "summary":      summary,
        "bullets":      bullets,
        "severity":     event.get("severity", "unknown"),
        "generated_at": datetime.utcnow().isoformat(),
        "llm_used":     llm_text is not None,
        "prompt_used":  prompt,
    }


@router.post("/full-pipeline")
def full_pipeline_advisory(body: dict):
    """
    Convenience: takes just { "flight_id": "...", "risk_threshold": 0.25 }
    Runs cascade + recovery + advisory in one shot.
    """
    flight_id      = body.get("flight_id")
    risk_threshold = body.get("risk_threshold", 0.25)

    if not flight_id:
        raise HTTPException(status_code=400, detail="flight_id required")

    # reuse the disruptions router logic
    import json, pickle
    from pathlib import Path
    from ml.feature_builder import build_historical_rates, build_downstream_index
    from ml.cascade_model   import predict_cascade
    from ml.severity_scorer import build_disruption_event
    from agents.coordinator import run_recovery_pipeline

    disruptions = json.loads(Path("data/processed/disruptions_seed.json").read_text())
    flights     = json.loads(Path("data/processed/flights.json").read_text())

    G = None
    gp = Path("data/processed/flight_graph.gpickle")
    if gp.exists():
        with open(gp, "rb") as f:
            G = pickle.load(f)

    disruption = next((d for d in disruptions if d["flight_id"] == flight_id), None)
    if not disruption:
        raise HTTPException(status_code=404, detail="No disruption found for flight_id")

    hist_rates     = build_historical_rates(flights)
    downstream_idx = build_downstream_index(flights)
    flights_lookup = {f["id"]: f for f in flights}

    candidates = [f for f in flights
                   if f["origin_id"] == disruption.get("destination", "")
                   and f["status"] not in ("cancelled",)][:50]

    affected = predict_cascade(disruption, candidates, G, hist_rates,
                                downstream_idx, risk_threshold=risk_threshold)
    event    = build_disruption_event(disruption, affected, flights_lookup, G)

    crew        = json.loads(Path("data/processed/crew_roster.json").read_text())
    fleet       = json.loads(Path("data/processed/aircraft_fleet.json").read_text())
    passengers  = json.loads(Path("data/processed/passengers.json").read_text())
    itineraries = json.loads(Path("data/processed/itineraries.json").read_text())

    affected_ids   = {a["flight_id"] for a in affected}
    affected_full  = [f for f in flights if f["id"] in affected_ids]

    plan = run_recovery_pipeline(
        disruption=event, affected_flights=affected_full,
        all_flights=flights, available_crew=crew,
        available_aircraft=fleet, passengers=passengers,
        itineraries=itineraries,
    )

    prompt   = _format_plan_for_llm(event, plan)
    llm_text = _call_groq(prompt)

    if llm_text:
        lines   = [l.strip() for l in llm_text.split("\n") if l.strip()]
        summary = lines[0] if lines else llm_text
        bullets = [l.lstrip("•-* ") for l in lines[1:]]
    else:
        summary, bullets = _rule_based_advisory(event, plan)

    return {
        "disruption_event": event,
        "recovery_plan":    plan,
        "advisory": {
            "summary":      summary,
            "bullets":      bullets,
            "severity":     event.get("severity"),
            "generated_at": datetime.utcnow().isoformat(),
            "llm_used":     llm_text is not None,
        },
    }