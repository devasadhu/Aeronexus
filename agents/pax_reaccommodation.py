"""
agents/pax_reaccommodation.py

Passenger Reaccommodation Engine — replaces the stub rebooking logic
in PassengerAgent with a proper graph-based search.

Algorithm:
  1. Build a time-expanded flight graph where nodes are (airport, time_bucket)
     and edges are flights (with capacity and delay cost weights).
  2. For each stranded/misconnecting passenger, run Dijkstra to find the
     lowest-cost path from current airport to final destination.
  3. Apply cabin-class inventory constraints and cost model.
  4. Return RebookingResult with itinerary, cost, and disruption delta.

Cost model (lower = better for passenger):
  - Each flight leg: delay_minutes * weight + connection_wait * 0.5
  - Overnight connection: +480 min penalty
  - Class downgrade: +120 min equivalent penalty
"""

from __future__ import annotations

import heapq
import logging
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
OVERNIGHT_PENALTY     = 480   # minutes — penalise solutions that strand pax overnight
CLASS_DOWNGRADE_PEN   = 120   # minutes equivalent — penalise cabin downgrade
MAX_CONNECTIONS       = 2     # max stops in rebooked itinerary
MAX_SEARCH_FLIGHTS    = 500   # cap graph size for speed
CONNECTION_MIN_GAP    = 45    # minimum connection time in minutes
CONNECTION_MAX_GAP    = 240   # maximum useful connection gap in minutes


# ── Data structures ───────────────────────────────────────────────────────────

class RebookingResult:
    def __init__(self, passenger_id: str, original_dest: str):
        self.passenger_id   = passenger_id
        self.original_dest  = original_dest
        self.itinerary: list[dict] = []   # list of flight dicts
        self.total_cost_min: float = 0.0  # lower = better
        self.total_delay_min: int  = 0
        self.connection_count: int = 0
        self.feasible: bool        = False
        self.reason: str           = "not attempted"

    def to_action(self, original_flight_number: str = "?") -> dict:
        legs = " → ".join(
            f"{l['origin']}→{l['dest']} ({l['flight_number']})"
            for l in self.itinerary
        )
        return {
            "action_type":   "passenger_rebook",
            "flight_id":     self.itinerary[0]["id"] if self.itinerary else "none",
            "flight_number": self.itinerary[0]["flight_number"] if self.itinerary else "—",
            "agent_source":  "passenger_reaccommodation",
            "feasible":      self.feasible,
            "conflict_flag": False,
            "description": (
                f"Rebook pax {self.passenger_id}: {legs} "
                f"(+{self.total_delay_min}min, {self.connection_count} connection(s))"
                if self.feasible
                else f"No viable reroute found for pax {self.passenger_id}: {self.reason}"
            ),
            "metadata": {
                "passenger_id":    self.passenger_id,
                "new_itinerary":   self.itinerary,
                "total_delay_min": self.total_delay_min,
                "connections":     self.connection_count,
                "cost_score":      round(self.total_cost_min, 1),
            },
        }


# ── Time-expanded graph ───────────────────────────────────────────────────────

def _parse_dt(s) -> Optional[datetime]:
    if isinstance(s, datetime):
        return s
    if isinstance(s, str):
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None


def _build_flight_index(flights: list) -> dict:
    """
    Index: origin_iata -> list of flights sorted by departure time.
    Only include scheduled/delayed flights with seats available.
    """
    index: dict[str, list] = {}
    for f in flights:
        if f.get("status") in ("cancelled", "arrived"):
            continue
        cap     = f.get("capacity", 150)
        booked  = f.get("booked_seats", 0)
        if booked >= cap:
            continue
        origin = f.get("origin_id", "")
        if origin:
            index.setdefault(origin, []).append(f)

    # sort each bucket by departure
    for origin in index:
        index[origin].sort(key=lambda f: f.get("scheduled_departure", ""))

    return index


# ── Dijkstra on time-expanded graph ──────────────────────────────────────────

def find_best_reroute(
    passenger_id:   str,
    stranded_at:    str,
    final_dest:     str,
    earliest_dep:   datetime,
    original_class: str,
    flights:        list,
    max_connections: int = MAX_CONNECTIONS,
) -> RebookingResult:
    """
    Dijkstra search for lowest-cost path from stranded_at to final_dest.

    State: (cost, airport, time, path_so_far, legs_used)
    """
    result = RebookingResult(passenger_id, final_dest)

    if stranded_at == final_dest:
        result.feasible        = True
        result.reason          = "already at destination"
        result.total_delay_min = 0
        return result

    flight_index = _build_flight_index(flights)

    # priority queue: (cost, airport, current_time_iso, path)
    heap = [(0.0, stranded_at, earliest_dep.isoformat(), [])]
    visited: dict[tuple, float] = {}   # (airport, time_bucket) -> best cost

    while heap:
        cost, airport, cur_time_iso, path = heapq.heappop(heap)

        if len(path) > max_connections:
            continue

        cur_time = _parse_dt(cur_time_iso)
        if cur_time is None:
            continue

        state_key = (airport, cur_time.strftime("%Y-%m-%dT%H"))
        if state_key in visited and visited[state_key] <= cost:
            continue
        visited[state_key] = cost

        # check neighbours
        for f in flight_index.get(airport, []):
            dep = _parse_dt(f.get("scheduled_departure"))
            arr = _parse_dt(f.get("scheduled_arrival"))
            if dep is None or arr is None:
                continue

            # must depart after current time + min connection gap
            gap_min = (dep - cur_time).total_seconds() / 60
            if gap_min < CONNECTION_MIN_GAP:
                continue
            if gap_min > CONNECTION_MAX_GAP and path:  # don't wait 4+ hours mid-journey
                continue

            dest = f.get("destination_id", "")
            if not dest:
                continue

            # cost: connection wait + flight duration
            flight_dur  = (arr - dep).total_seconds() / 60
            wait_cost   = gap_min * 0.5
            dur_cost    = flight_dur

            # overnight penalty
            if gap_min > 360:
                wait_cost += OVERNIGHT_PENALTY

            # cabin downgrade penalty (simplified: economy only available)
            if original_class in ("business", "first"):
                wait_cost += CLASS_DOWNGRADE_PEN

            edge_cost = wait_cost + dur_cost
            new_cost  = cost + edge_cost

            new_leg = {
                "id":            f["id"],
                "flight_number": f.get("flight_number", ""),
                "origin":        airport,
                "dest":          dest,
                "departure":     dep.isoformat(),
                "arrival":       arr.isoformat(),
                "avail_seats":   f.get("capacity", 150) - f.get("booked_seats", 0),
            }
            new_path = path + [new_leg]

            if dest == final_dest:
                # found a route
                total_delay = int((arr - earliest_dep).total_seconds() / 60)
                if not result.feasible or new_cost < result.total_cost_min:
                    result.feasible          = True
                    result.itinerary         = new_path
                    result.total_cost_min    = new_cost
                    result.total_delay_min   = max(0, total_delay)
                    result.connection_count  = len(new_path) - 1
                    result.reason            = "reroute found"
            else:
                heapq.heappush(heap, (new_cost, dest, arr.isoformat(), new_path))

    if not result.feasible:
        result.reason = f"no viable route from {stranded_at} to {final_dest} found"

    return result


# ── Batch reaccommodation ─────────────────────────────────────────────────────

def reaccommodate_passengers(
    disruption_event:    dict,
    affected_flights:    list,
    all_flights:         list,
    passengers:          list,
    itineraries:         list,
    cancelled_flight_ids: set,
) -> list[dict]:
    """
    Main entry point called by CoordinatorAgent or PassengerAgent.

    Returns list of action dicts (passenger_rebook).
    """
    from datetime import timezone

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # build passenger lookup
    pax_by_id  = {p["id"]: p for p in passengers}
    itin_by_id = {}
    for it in itineraries:
        pid = it.get("passenger_id")
        itin_by_id.setdefault(pid, []).append(it)

    # disrupted flight details
    disrupted_origins = {f.get("origin_id") for f in affected_flights}
    disrupted_dests   = {f.get("destination_id") for f in affected_flights}
    disrupted_ids     = {f["id"] for f in affected_flights} | cancelled_flight_ids

    actions = []
    processed = 0

    for pax in passengers[:100]:   # cap for performance
        pid  = pax["id"]
        itins = itin_by_id.get(pid, [])

        # find itinerary legs that touch a disrupted flight
        affected_legs = [it for it in itins if it.get("flight_id") in disrupted_ids]
        if not affected_legs:
            continue

        # take the first affected leg
        leg          = affected_legs[0]
        stranded_at  = leg.get("origin", "")
        final_dest   = leg.get("destination", "")
        orig_class   = pax.get("cabin_class", "economy")

        if not stranded_at or not final_dest:
            continue

        # earliest they can depart = now + 30 min (assume at gate)
        earliest_dep = now + timedelta(minutes=30)

        result = find_best_reroute(
            passenger_id   = pid,
            stranded_at    = stranded_at,
            final_dest     = final_dest,
            earliest_dep   = earliest_dep,
            original_class = orig_class,
            flights        = all_flights,
            max_connections= MAX_CONNECTIONS,
        )

        actions.append(result.to_action(leg.get("flight_number", "?")))
        processed += 1

    log.info(
        f"[PaxReaccommodation] processed={processed} | "
        f"rebooked={sum(1 for a in actions if a['feasible'])} | "
        f"stranded={sum(1 for a in actions if not a['feasible'])}"
    )
    return actions


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    flights     = json.loads(Path("data/processed/flights.json").read_text())
    disruptions = json.loads(Path("data/processed/disruptions_seed.json").read_text())
    passengers  = json.loads(Path("data/processed/passengers.json").read_text())
    itineraries = json.loads(Path("data/processed/itineraries.json").read_text())

    # use first disruption's destination as stranded location
    d           = disruptions[0]
    stranded_at = d["destination"]
    final_dest  = "JFK" if stranded_at != "JFK" else "LAX"

    from datetime import timezone
    result = find_best_reroute(
        passenger_id   = "demo_pax_001",
        stranded_at    = stranded_at,
        final_dest     = final_dest,
        earliest_dep   = datetime.now(timezone.utc).replace(tzinfo=None),
        original_class = "economy",
        flights        = flights,
    )

    log.info(f"Feasible: {result.feasible}")
    log.info(f"Delay: {result.total_delay_min} min | Connections: {result.connection_count}")
    for leg in result.itinerary:
        log.info(f"  {leg['origin']} → {leg['dest']} on {leg['flight_number']}")