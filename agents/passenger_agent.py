"""
agents/passenger_agent.py

PassengerAgent: proposes rebooking plans for disrupted passengers.

Strategy (two-tier):
  Tier 1 — Same-route fast rebook (unchanged logic):
    Find another flight on the same origin→dest within 6h with available seats.
    Sorted by departure time, highest-priority passengers first.

  Tier 2 — Graph-based reroute (via pax_reaccommodation engine):
    If Tier 1 finds no seat (stranded), hand off to the Dijkstra engine in
    pax_reaccommodation.py which searches the full flight network for a
    multi-hop path to the passenger's final destination.

Scoring (Tier 1 priority, higher = more urgent):
  - fare_class:      first=30, business=20, economy=10
  - frequent_flyer:  +5
  - misconnect_risk: * 8
  - connection_time: <45min +6, <90min +3
"""

import sys, logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
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
    Tier 1: same origin→dest, within window_hours, with capacity.
    Returns list of (flight_dict, remaining_seats, departure_dt).
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

    alts.sort(key=lambda x: x[2])
    return alts


class PassengerAgent:

    def run(self, context: dict) -> dict:
        """
        context keys:
          disruption          - DisruptionEvent dict
          affected_flights    - list of disrupted flight dicts
          all_flights         - full flight list
          passengers          - list of passenger dicts
          itineraries         - list of itinerary dicts
          booked_seats_delta  - {flight_id: seats_already_committed}

        Returns AgentProposal dict.
        """
        affected_flights     = context["affected_flights"]
        all_flights          = context.get("all_flights", [])
        passengers           = context.get("passengers", [])
        itineraries          = context.get("itineraries", [])
        booked_delta         = context.get("booked_seats_delta", {})

        disrupted_flight_ids = {f["id"] for f in affected_flights}

        pax_by_id      = {p["id"]: p for p in passengers}
        itin_by_flight = defaultdict(list)
        for i in itineraries:
            itin_by_flight[i["flight_id"]].append(i)

        seats_committed = dict(booked_delta)

        actions          = []
        rebooked_count   = 0
        rerouted_count   = 0
        stranded_count   = 0
        total_cost       = 0.0

        # collect passengers stranded after Tier 1 for Tier 2 batch call
        stranded_pax: list[dict] = []   # {passenger, itin, flight}

        # ── Tier 1: same-route fast rebook ───────────────────────────────────
        for flight in affected_flights:
            flight_id = flight["id"]
            itins     = itin_by_flight.get(flight_id, [])
            if not itins:
                continue

            pax_queue = []
            for itin in itins:
                pax = pax_by_id.get(itin["passenger_id"])
                if not pax:
                    continue
                pax_queue.append((_priority_score(pax, itin), pax, itin))
            pax_queue.sort(key=lambda x: x[0], reverse=True)

            alts = _find_alternatives(flight, all_flights, disrupted_flight_ids)

            for priority, pax, itin in pax_queue:
                rebooked = False

                for alt_flight, remaining, alt_dep in alts:
                    alt_id           = alt_flight["id"]
                    committed_so_far = seats_committed.get(alt_id, 0)
                    avail            = remaining - committed_so_far
                    if avail <= 0:
                        continue

                    seats_committed[alt_id] = committed_so_far + 1
                    cost = 0.05 if pax.get("fare_class") == "economy" else 0.02
                    total_cost   += cost
                    rebooked_count += 1
                    rebooked = True

                    dep_orig  = _parse_dt(flight.get("scheduled_departure"))
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
                            "pnr":               pax["pnr"],
                            "fare_class":        pax["fare_class"],
                            "frequent_flyer":    pax.get("frequent_flyer", False),
                            "original_flight":   flight.get("flight_number", ""),
                            "new_flight_id":     alt_id,
                            "new_flight_number": alt_flight.get("flight_number", ""),
                            "delay_minutes":     delay_min,
                            "misconnect_risk":   itin.get("misconnect_risk", 0),
                            "priority_score":    round(priority, 2),
                            "rebook_tier":       "same_route",
                        },
                    })
                    break

                if not rebooked:
                    # queue for Tier 2
                    stranded_pax.append({
                        "passenger": pax,
                        "itin":      itin,
                        "flight":    flight,
                        "priority":  priority,
                    })

        # ── Tier 2: graph-based reroute for stranded passengers ───────────────
        if stranded_pax:
            try:
                from agents.pax_reaccommodation import find_best_reroute

                now = datetime.now(timezone.utc).replace(tzinfo=None)

                for item in stranded_pax:
                    pax      = item["passenger"]
                    itin     = item["itin"]
                    flight   = item["flight"]
                    priority = item["priority"]

                    stranded_at  = flight.get("origin_id", "")
                    final_dest   = flight.get("destination_id", "")
                    orig_class   = pax.get("cabin_class", pax.get("fare_class", "economy"))

                    if not stranded_at or not final_dest:
                        stranded_count += 1
                        continue

                    earliest_dep = now + timedelta(minutes=30)

                    result = find_best_reroute(
                        passenger_id    = pax["id"],
                        stranded_at     = stranded_at,
                        final_dest      = final_dest,
                        earliest_dep    = earliest_dep,
                        original_class  = orig_class,
                        flights         = all_flights,
                    )

                    if result.feasible:
                        rerouted_count += 1
                        total_cost     += 0.15   # reroute cost weight
                        action          = result.to_action(flight.get("flight_number", "?"))
                        # enrich with passenger PNR for display
                        action["passenger_id"] = pax["id"]
                        action["flight_id"]    = flight["id"]
                        action["flight_number"]= flight.get("flight_number", "")
                        action["impact_score"] = priority / 30
                        action["cost_score"]   = 0.15
                        action["metadata"]["pnr"]          = pax.get("pnr", "")
                        action["metadata"]["fare_class"]   = pax.get("fare_class", "economy")
                        action["metadata"]["rebook_tier"]  = "graph_reroute"
                        action["metadata"]["misconnect_risk"] = itin.get("misconnect_risk", 0)
                        actions.append(action)
                    else:
                        stranded_count += 1
                        actions.append({
                            "action_type":   "passenger_rebook",
                            "agent_source":  "passenger",
                            "flight_id":     flight["id"],
                            "flight_number": flight.get("flight_number", ""),
                            "passenger_id":  pax["id"],
                            "description": (
                                f"STRANDED: PNR {pax.get('pnr','?')} ({pax['name']}, "
                                f"{pax.get('fare_class','economy')}) — "
                                f"no route found {stranded_at}→{final_dest}. "
                                f"Manual handling required."
                            ),
                            "cost_score":    0.8,
                            "impact_score":  priority / 30,
                            "feasible":      False,
                            "conflict_flag": False,
                            "metadata": {
                                "pnr":          pax.get("pnr", ""),
                                "fare_class":   pax.get("fare_class", "economy"),
                                "frequent_flyer": pax.get("frequent_flyer", False),
                                "reason":       result.reason,
                                "rebook_tier":  "graph_reroute_failed",
                            },
                        })

            except ImportError:
                log.warning(
                    "[PassengerAgent] pax_reaccommodation not found — "
                    "marking all Tier-2 passengers as stranded."
                )
                for item in stranded_pax:
                    pax      = item["passenger"]
                    flight   = item["flight"]
                    priority = item["priority"]
                    stranded_count += 1
                    actions.append({
                        "action_type":   "passenger_rebook",
                        "agent_source":  "passenger",
                        "flight_id":     flight["id"],
                        "flight_number": flight.get("flight_number", ""),
                        "passenger_id":  pax["id"],
                        "description": (
                            f"STRANDED: PNR {pax.get('pnr','?')} ({pax['name']}, "
                            f"{pax.get('fare_class','economy')}) — "
                            f"no alternative on same route, reroute engine unavailable. "
                            f"Manual handling required."
                        ),
                        "cost_score":    0.8,
                        "impact_score":  priority / 30,
                        "feasible":      False,
                        "conflict_flag": False,
                        "metadata": {
                            "pnr":        pax.get("pnr", ""),
                            "fare_class": pax.get("fare_class", "economy"),
                            "reason":     "reroute_engine_unavailable",
                        },
                    })

        # ── Misconnect summary ────────────────────────────────────────────────
        high_risk_itins = [
            i for i in itineraries
            if i["flight_id"] in disrupted_flight_ids
            and i.get("misconnect_risk", 0) > 0.5
        ]

        proposal = {
            "agent":      "passenger",
            "actions":    actions,
            "cost":       round(total_cost, 3),
            "confidence": 0.75,
            "notes": (
                f"Same-route rebooked: {rebooked_count}. "
                f"Graph-rerouted: {rerouted_count}. "
                f"Stranded (manual): {stranded_count}. "
                f"High misconnect risk: {len(high_risk_itins)}."
            ),
            "metadata": {
                "rebooked_count":    rebooked_count,
                "rerouted_count":    rerouted_count,
                "stranded_count":    stranded_count,
                "high_risk_count":   len(high_risk_itins),
                "seats_committed":   seats_committed,
            },
        }

        log.info(
            f"[PassengerAgent] same_route={rebooked_count} "
            f"graph_reroute={rerouted_count} "
            f"stranded={stranded_count} "
            f"high_risk={len(high_risk_itins)} | cost={total_cost:.2f}"
        )
        return proposal