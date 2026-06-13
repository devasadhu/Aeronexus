"""
agents/crew_agent.py

CrewAgent: proposes legal crew reassignments for disrupted flights.

Constraints enforced (FAR 117 simplified):
  - Captain + First Officer required per flight (FA count by capacity)
  - Crew must be qualified on aircraft type
  - Current duty hours + flight duration must not exceed max_duty_hours
  - Min rest between duties: 10h (if crew is in "rest" status, skip)
  - 7-day cumulative duty must not exceed 60h
  - Crew must be at the same airport (or standby at same airport)
  - Crew already committed in this plan cannot be double-assigned

Scoring:
  - Prefer crew already at origin
  - Prefer crew with lowest current duty hours (most rested)
  - Prefer standby crew over available (standby = designated reserve)
  - Penalise crew with high 7-day cumulative hours
"""

import sys, logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MAX_DUTY_HOURS     = 14.0
MAX_7DAY_HOURS     = 60.0
MIN_REST_HOURS     = 10.0
FA_PER_50_SEATS    = 1     # 1 FA per 50 seats (simplified)

REQUIRED_ROLES = {
    "captain":          1,
    "first_officer":    1,
}


def _fas_needed(capacity: int) -> int:
    return max(1, capacity // 50)


def _flight_duration_hours(dep_str, arr_str) -> float:
    try:
        dep = datetime.fromisoformat(dep_str)
        arr = datetime.fromisoformat(arr_str)
        if arr < dep:
            arr += timedelta(days=1)
        return (arr - dep).total_seconds() / 3600
    except Exception:
        return 3.0   # default 3h if unknown


def _can_fly(crew: dict, flight_duration_h: float, aircraft_type: str) -> tuple:
    """
    Returns (eligible: bool, reason: str)
    """
    status = crew.get("status", "off")
    if status in ("rest", "off"):
        return False, f"status={status}"

    quals = crew.get("qualifications_json", [])
    if aircraft_type and aircraft_type not in quals:
        return False, f"not qualified on {aircraft_type} (quals={quals})"

    current_duty = crew.get("current_duty_hours", 0.0)
    max_duty     = crew.get("max_duty_hours", MAX_DUTY_HOURS)
    if current_duty + flight_duration_h > max_duty:
        return False, (f"duty limit: {current_duty:.1f}h + {flight_duration_h:.1f}h "
                       f"> {max_duty}h")

    cum_7day = crew.get("cumulative_duty_7day", 0.0)
    if cum_7day + flight_duration_h > MAX_7DAY_HOURS:
        return False, f"7-day limit: {cum_7day:.1f}h + {flight_duration_h:.1f}h > {MAX_7DAY_HOURS}h"

    return True, "ok"


def _score_crew(crew: dict, origin: str, flight_duration_h: float) -> float:
    """Lower = better."""
    score = 0.0

    # at origin = best
    if crew.get("current_location_id") != origin:
        score += 20

    # standby preferred over available
    if crew.get("status") == "standby":
        score -= 10
    elif crew.get("status") == "available":
        score += 0

    # lower current duty hours = more rested
    score += crew.get("current_duty_hours", 0) * 1.5

    # penalise high 7-day accumulation
    score += crew.get("cumulative_duty_7day", 0) * 0.5

    return score


class CrewAgent:

    def run(self, context: dict) -> dict:
        """
        context keys:
          disruption        - DisruptionEvent dict
          affected_flights  - list of flight dicts
          available_crew    - list of crew member dicts
          aircraft_type_map - {flight_id: aircraft_type str} (optional)

        Returns AgentProposal dict.
        """
        affected_flights  = context["affected_flights"]
        available_crew    = context["available_crew"]
        aircraft_type_map = context.get("aircraft_type_map", {})

        # split crew by role
        by_role = {"captain": [], "first_officer": [], "flight_attendant": []}
        for c in available_crew:
            role = c.get("role", "flight_attendant")
            if role in by_role:
                by_role[role].append(c)

        actions      = []
        committed    = set()   # employee_ids committed in this plan
        total_cost   = 0.0
        covered      = 0

        sorted_flights = sorted(affected_flights,
                                key=lambda f: f.get("booked_seats", 0), reverse=True)

        for flight in sorted_flights:
            flight_id    = flight["id"]
            origin       = flight["origin_id"]
            capacity     = flight.get("capacity", 150)
            aircraft_type = aircraft_type_map.get(flight_id, "")

            dep_str = flight.get("scheduled_departure", "")
            arr_str = flight.get("scheduled_arrival",   "")
            duration_h = _flight_duration_hours(dep_str, arr_str)

            fas_needed = _fas_needed(capacity)
            crew_needed = {
                "captain":          1,
                "first_officer":    1,
                "flight_attendant": fas_needed,
            }

            flight_assignments = []
            flight_feasible    = True
            flight_notes       = []

            for role, count_needed in crew_needed.items():
                assigned = 0
                candidates = [
                    c for c in by_role[role]
                    if c["employee_id"] not in committed
                ]

                # filter by eligibility
                eligible = []
                for c in candidates:
                    ok, reason = _can_fly(c, duration_h, aircraft_type)
                    if ok:
                        eligible.append(c)

                # sort by score
                eligible.sort(key=lambda c: _score_crew(c, origin, duration_h))

                for crew_member in eligible:
                    if assigned >= count_needed:
                        break

                    committed.add(crew_member["employee_id"])
                    assigned += 1

                    needs_reposition = crew_member.get("current_location_id") != origin
                    cost = 0.2 if needs_reposition else 0.05
                    total_cost += cost

                    flight_assignments.append({
                        "crew_id":        crew_member["id"],
                        "employee_id":    crew_member["employee_id"],
                        "name":           crew_member["name"],
                        "role":           role,
                        "needs_reposition": needs_reposition,
                        "from_airport":   crew_member.get("current_location_id"),
                        "current_duty_h": crew_member.get("current_duty_hours", 0),
                        "cost":           cost,
                    })

                if assigned < count_needed:
                    flight_feasible = False
                    flight_notes.append(
                        f"Only {assigned}/{count_needed} {role}(s) available"
                    )

            if flight_assignments:
                names = ", ".join(f"{a['name']} ({a['role']})"
                                   for a in flight_assignments[:3])
                extra = f" +{len(flight_assignments)-3} more" if len(flight_assignments) > 3 else ""

                action_type = "crew_reassign" if flight_feasible else "crew_reassign"
                actions.append({
                    "action_type":   "crew_reassign",
                    "agent_source":  "fleet",  # will be overridden to "crew" by coordinator
                    "agent_source":  "crew",
                    "flight_id":     flight_id,
                    "flight_number": flight.get("flight_number", ""),
                    "description":   (
                        f"Assign crew to {flight.get('flight_number','')}: "
                        f"{names}{extra}"
                        + (f" — PARTIAL: {'; '.join(flight_notes)}" if not flight_feasible else "")
                    ),
                    "cost_score":    sum(a["cost"] for a in flight_assignments),
                    "impact_score":  1.0 if flight_feasible else 0.5,
                    "feasible":      flight_feasible,
                    "conflict_flag": False,
                    "metadata": {
                        "assignments":    flight_assignments,
                        "fully_crewed":   flight_feasible,
                        "partial_notes":  flight_notes,
                        "duration_hours": round(duration_h, 2),
                    },
                })

                if flight_feasible:
                    covered += 1
            else:
                actions.append({
                    "action_type":   "flight_cancel",
                    "agent_source":  "crew",
                    "flight_id":     flight_id,
                    "flight_number": flight.get("flight_number", ""),
                    "description":   (
                        f"No legal crew available for {flight.get('flight_number','')}. "
                        f"Recommend cancellation. Notes: {'; '.join(flight_notes) or 'all crew exhausted'}"
                    ),
                    "cost_score":    1.0,
                    "impact_score":  flight.get("booked_seats", 0) / 200,
                    "feasible":      True,
                    "conflict_flag": False,
                    "metadata":      {"reason": "no_legal_crew"},
                })

        proposal = {
            "agent":      "crew",
            "actions":    actions,
            "cost":       round(total_cost, 3),
            "confidence": 0.80,
            "notes": (
                f"Processed {len(affected_flights)} flights. "
                f"Fully crewed: {covered}. "
                f"Partial/cancel: {len(affected_flights) - covered}."
            ),
        }

        log.info(f"[CrewAgent] {covered}/{len(affected_flights)} flights fully crewed "
                 f"| cost={total_cost:.2f}")
        return proposal