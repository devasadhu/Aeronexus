"""
agents/passenger_agent.py

PassengerAgent: proposes rebooking plans for disrupted passengers.

Logic:
  1. Identify passengers on disrupted flights (from itineraries)
  2. Find alternative flights on same route with available capacity
  3. Prioritise by: fare class > frequent flyer > misconnect risk > connection time
  4. Assign to best available alternative (earliest departure, sufficient capacity)
  5. Flag passengers who cannot be rebooked for manual handling

Scoring:
  - Prefer same-day alternatives
  - Prefer direct routes over connections
  - Penalise alternatives with low remaining capacity
"""

import sys, logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

FARE_PRIORITY = {"first": 3, "business": 2, "economy": 1}


def _parse_dt(s) -> datetime:
    if isinstance(s, datetime):
        return s
    try:
        return datetime.fromisoformat(str(s))
    except Exception:
        return datetime.utcnow()


def _priority_score(passenger: dict, itinerary: dict) -> float:
    """Higher = more urgent to rebook."""
    score = 0.0
    score += FARE_PRIORITY.get(passenger.get("fare_class", "economy"), 1) * 10
    if passenger.get("frequent_flyer"):
        score += 5
    score += itinerary.get("misconnect_risk", 0) * 8
    conn_time = itinerary.get("connection_time_min") or 999
    if conn_time < 45:
        score += 6
    elif conn_time < 90:
        score += 3
    return score


def _find_alternatives(
    flight: dict,
    all_flights: list,
    disrupted_flight_ids: set,
    window_hours: int = 6,
) -> list:
    """
    Find flights on the same route departing within window_hours after
    the disrupted flight's scheduled departure.
    Excludes already-disrupted flights.
    """
    origin = flight["origin_id"]
    dest   = flight["destination_id"]
    dep    = _parse_dt(flight.get("scheduled_departure"))
    window_end = dep + timedelta(hours=window_hours)

    alts = []
    for f in all_flights:
        if f["id"] in disrupted_flight_ids:
            continue
        if f["origin_id"] != origin or f["destination_id"] != dest:
            continue
        if f["status"] in ("cancelled", "diverted"):
            continue
        f_dep = _parse_dt(f.get("scheduled_departure"))
        if dep <= f_dep <= window_end:
            remaining = f.get("capacity", 150) - f.get("booked_seats", 0)
            if remaining > 0:
                alts.append((f, remaining, f_dep))

    # sort by departure time
    alts.sort(key=lambda x: x[2])
    return alts


class PassengerAgent:

    def run(self, context: dict) -> dict:
        """
        context keys:
          disruption          - DisruptionEvent dict
          affected_flights    - list of disrupted flight dicts
          all_flights         - full flight list for finding alternatives
          passengers          - list of passenger dicts
          itineraries         - list of itinerary dicts
          booked_seats_delta  - {flight_id: seats_already_committed} (from prior runs)

        Returns AgentProposal dict.
        """
        affected_flights    = context["affected_flights"]
        all_flights         = context.get("all_flights", [])
        passengers          = context.get("passengers", [])
        itineraries         = context.get("itineraries", [])
        booked_delta        = context.get("booked_seats_delta", {})

        disrupted_flight_ids = {f["id"] for f in affected_flights}

        # index lookups
        pax_by_id   = {p["id"]: p for p in passengers}
        itin_by_flight = defaultdict(list)
        for i in itineraries:
            itin_by_flight[i["flight_id"]].append(i)

        # track seats committed in this plan: {flight_id: seats_booked}
        seats_committed = dict(booked_delta)

        actions = []
        rebooked_count   = 0
        stranded_count   = 0
        total_cost       = 0.0

        for flight in affected_flights:
            flight_id  = flight["id"]
            itins      = itin_by_flight.get(flight_id, [])

            if not itins:
                continue

            # build (passenger, itinerary, priority) triples
            pax_queue = []
            for itin in itins:
                pax = pax_by_id.get(itin["passenger_id"])
                if not pax:
                    continue
                priority = _priority_score(pax, itin)
                pax_queue.append((priority, pax, itin))

            # sort highest priority first
            pax_queue.sort(key=lambda x: x[0], reverse=True)

            # find alternatives for this flight
            alts = _find_alternatives(flight, all_flights, disrupted_flight_ids)

            for priority, pax, itin in pax_queue:
                rebooked = False

                for alt_flight, remaining, alt_dep in alts:
                    alt_id = alt_flight["id"]
                    committed_so_far = seats_committed.get(alt_id, 0)
                    avail = remaining - committed_so_far

                    if avail <= 0:
                        continue

                    seats_committed[alt_id] = committed_so_far + 1
                    cost = 0.05 if pax.get("fare_class") == "economy" else 0.02
                    total_cost += cost
                    rebooked_count += 1
                    rebooked = True

                    dep_orig = _parse_dt(flight.get("scheduled_departure"))
                    delay_min = int((alt_dep - dep_orig).total_seconds() / 60)

                    actions.append({
                        "action_type":   "passenger_rebook",
                        "agent_source":  "passenger",
                        "flight_id":     flight_id,
                        "flight_number": flight.get("flight_number", ""),
                        "passenger_id":  pax["id"],
                        "description": (
                            f"Rebook PNR {pax['pnr']} ({pax['name']}, {pax['fare_class']}) "
                            f"from {flight.get('flight_number','')} → "
                            f"{alt_flight.get('flight_number','')} "
                            f"(+{delay_min}min delay)"
                        ),
                        "cost_score":    cost,
                        "impact_score":  priority / 30,
                        "feasible":      True,
                        "conflict_flag": False,
                        "metadata": {
                            "pnr":              pax["pnr"],
                            "fare_class":       pax["fare_class"],
                            "frequent_flyer":   pax.get("frequent_flyer", False),
                            "original_flight":  flight.get("flight_number", ""),
                            "new_flight_id":    alt_id,
                            "new_flight_number": alt_flight.get("flight_number", ""),
                            "delay_minutes":    delay_min,
                            "misconnect_risk":  itin.get("misconnect_risk", 0),
                            "priority_score":   round(priority, 2),
                        },
                    })
                    break   # rebooked — move to next passenger

                if not rebooked:
                    stranded_count += 1
                    actions.append({
                        "action_type":   "passenger_rebook",
                        "agent_source":  "passenger",
                        "flight_id":     flight_id,
                        "flight_number": flight.get("flight_number", ""),
                        "passenger_id":  pax["id"],
                        "description": (
                            f"STRANDED: PNR {pax['pnr']} ({pax['name']}, {pax['fare_class']}) "
                            f"— no alternative found within 6h on "
                            f"{flight.get('origin_id','?')}→{flight.get('destination_id','?')}. "
                            f"Manual handling required."
                        ),
                        "cost_score":    0.8,
                        "impact_score":  priority / 30,
                        "feasible":      False,
                        "conflict_flag": False,
                        "metadata": {
                            "pnr":            pax["pnr"],
                            "fare_class":     pax["fare_class"],
                            "frequent_flyer": pax.get("frequent_flyer", False),
                            "reason":         "no_alternative_capacity",
                        },
                    })

        # misconnect risk summary
        high_risk_itins = [i for i in itineraries
                            if i["flight_id"] in disrupted_flight_ids
                            and i.get("misconnect_risk", 0) > 0.5]

        proposal = {
            "agent":      "passenger",
            "actions":    actions,
            "cost":       round(total_cost, 3),
            "confidence": 0.75,
            "notes": (
                f"Rebooked: {rebooked_count}. "
                f"Stranded (manual): {stranded_count}. "
                f"High misconnect risk itins: {len(high_risk_itins)}."
            ),
            "metadata": {
                "rebooked_count":    rebooked_count,
                "stranded_count":    stranded_count,
                "high_risk_count":   len(high_risk_itins),
                "seats_committed":   seats_committed,
            },
        }

        log.info(f"[PassengerAgent] rebooked={rebooked_count} stranded={stranded_count} "
                 f"high_risk_itins={len(high_risk_itins)} | cost={total_cost:.2f}")
        return proposal