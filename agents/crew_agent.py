"""
agents/crew_agent.py

CrewAgent: proposes legal crew reassignments for disrupted flights.

Now uses crew_legality.py for full FAR 117 compliance checks:
  - FDP limit from Table B (report hour × segment count)
  - Minimum rest (10h standard, 8h reduced)
  - 7-day cumulative duty limit (60h)
  - 28-day cumulative duty limit (100h)
  - 365-day flight time limit (1000h)
  - Type qualification check
  - Status check (rest/off = ineligible)

Scoring (lower = better candidate):
  - At origin airport: −10
  - Standby status: −5
  - Remaining FDP margin: higher margin = lower score
  - Current duty hours: penalised

The legality verdict is attached to each assignment in metadata so the
dashboard can show exactly why each crew member was accepted or rejected.
"""

import sys, logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

FA_PER_50_SEATS = 1


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
        return 3.0


def _score_crew(crew: dict, origin: str, remaining_fdp_min: float) -> float:
    """Lower = better candidate."""
    score = 0.0
    if crew.get("current_location_id") != origin:
        score += 20
    if crew.get("status") == "standby":
        score -= 5
    # prefer most FDP margin remaining
    score -= remaining_fdp_min * 0.01
    score += crew.get("current_duty_hours", 0) * 1.5
    score += crew.get("cumulative_duty_7day", 0) * 0.3
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
        try:
            from agents.crew_legality import filter_legal_crew, check_legality
            legality_available = True
        except ImportError:
            log.warning("[CrewAgent] crew_legality not found — falling back to basic checks")
            legality_available = False

        affected_flights  = context["affected_flights"]
        available_crew    = context["available_crew"]
        aircraft_type_map = context.get("aircraft_type_map", {})

        # split crew by role
        by_role = {"captain": [], "first_officer": [], "flight_attendant": []}
        for c in available_crew:
            role = c.get("role", "flight_attendant")
            if role in by_role:
                by_role[role].append(c)

        actions       = []
        committed     = set()   # employee_ids committed in this plan
        total_cost    = 0.0
        covered       = 0

        # sort by passenger load — most impactful flights first
        sorted_flights = sorted(
            affected_flights,
            key=lambda f: f.get("booked_seats", 0),
            reverse=True
        )

        for flight in sorted_flights:
            flight_id     = flight["id"]
            origin        = flight["origin_id"]
            capacity      = flight.get("capacity", 150)
            aircraft_type = aircraft_type_map.get(flight_id, "")
            dep_str       = flight.get("scheduled_departure", "")
            arr_str       = flight.get("scheduled_arrival",   "")
            duration_h    = _flight_duration_hours(dep_str, arr_str)
            fas_needed    = _fas_needed(capacity)

            crew_needed = {
                "captain":          1,
                "first_officer":    1,
                "flight_attendant": fas_needed,
            }

            flight_assignments = []
            flight_feasible    = True
            flight_notes       = []
            rejection_log      = []   # for dashboard explainability

            for role, count_needed in crew_needed.items():
                candidates = [
                    c for c in by_role[role]
                    if c["employee_id"] not in committed
                ]

                if legality_available:
                    # Full FAR 117 filter — returns (crew, verdict) pairs
                    legal_pairs = filter_legal_crew(
                        candidates, flight, aircraft_type, num_segments=1
                    )

                    # log rejections for explainability
                    legal_ids = {c["employee_id"] for c, _ in legal_pairs}
                    for c in candidates:
                        if c["employee_id"] not in legal_ids:
                            v = check_legality(c, flight, aircraft_type)
                            rejection_log.append({
                                "name":       c["name"],
                                "role":       role,
                                "violations": v.violations,
                                "warnings":   v.warnings,
                            })

                    # sort by score (most margin, at origin, standby first)
                    legal_pairs.sort(
                        key=lambda pair: _score_crew(
                            pair[0], origin, pair[1].remaining_duty_min
                        )
                    )
                    eligible = [(c, v) for c, v in legal_pairs]

                else:
                    # Fallback: basic status + duty check
                    eligible = []
                    for c in candidates:
                        status = c.get("status", "off")
                        if status in ("rest", "off"):
                            continue
                        current_duty = c.get("current_duty_hours", 0.0)
                        max_duty     = c.get("max_duty_hours", 14.0)
                        if current_duty + duration_h > max_duty:
                            continue
                        eligible.append((c, None))
                    eligible.sort(
                        key=lambda pair: _score_crew(pair[0], origin, 0)
                    )

                assigned = 0
                for crew_member, verdict in eligible:
                    if assigned >= count_needed:
                        break

                    committed.add(crew_member["employee_id"])
                    assigned += 1

                    needs_reposition = crew_member.get("current_location_id") != origin
                    cost = 0.2 if needs_reposition else 0.05
                    total_cost += cost

                    assign_entry = {
                        "crew_id":          crew_member["id"],
                        "employee_id":      crew_member["employee_id"],
                        "name":             crew_member["name"],
                        "role":             role,
                        "needs_reposition": needs_reposition,
                        "from_airport":     crew_member.get("current_location_id"),
                        "current_duty_h":   crew_member.get("current_duty_hours", 0),
                        "cost":             cost,
                    }

                    # attach legality detail if available
                    if verdict is not None:
                        assign_entry["legality"] = {
                            "legal":               verdict.legal,
                            "max_fdp_min":         verdict.max_fdp_min,
                            "used_fdp_min":        round(verdict.used_fdp_min, 1),
                            "remaining_fdp_min":   round(verdict.remaining_duty_min, 1),
                            "warnings":            verdict.warnings,
                        }

                    flight_assignments.append(assign_entry)

                if assigned < count_needed:
                    flight_feasible = False
                    flight_notes.append(
                        f"Only {assigned}/{count_needed} legal {role}(s) available"
                    )

            if flight_assignments:
                names = ", ".join(
                    f"{a['name']} ({a['role']})" for a in flight_assignments[:3]
                )
                extra = (
                    f" +{len(flight_assignments)-3} more"
                    if len(flight_assignments) > 3 else ""
                )
                actions.append({
                    "action_type":   "crew_reassign",
                    "agent_source":  "crew",
                    "flight_id":     flight_id,
                    "flight_number": flight.get("flight_number", ""),
                    "description": (
                        f"Assign crew to {flight.get('flight_number','')}: "
                        f"{names}{extra}"
                        + (f" — PARTIAL: {'; '.join(flight_notes)}"
                           if not flight_feasible else "")
                    ),
                    "cost_score":    sum(a["cost"] for a in flight_assignments),
                    "impact_score":  1.0 if flight_feasible else 0.5,
                    "feasible":      flight_feasible,
                    "conflict_flag": False,
                    "metadata": {
                        "assignments":      flight_assignments,
                        "fully_crewed":     flight_feasible,
                        "partial_notes":    flight_notes,
                        "duration_hours":   round(duration_h, 2),
                        "aircraft_type":    aircraft_type,
                        "far117_checked":   legality_available,
                        "rejected_crew":    rejection_log,   # explainability
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
                    "description": (
                        f"No legal crew for {flight.get('flight_number','')}. "
                        f"Recommend cancellation. "
                        f"{'; '.join(flight_notes) or 'all crew exhausted or illegal'}"
                    ),
                    "cost_score":    1.0,
                    "impact_score":  flight.get("booked_seats", 0) / 200,
                    "feasible":      True,
                    "conflict_flag": False,
                    "metadata": {
                        "reason":         "no_legal_crew",
                        "rejected_crew":  rejection_log,
                        "far117_checked": legality_available,
                    },
                })

        proposal = {
            "agent":      "crew",
            "actions":    actions,
            "cost":       round(total_cost, 3),
            "confidence": 0.85 if legality_available else 0.70,
            "notes": (
                f"Processed {len(affected_flights)} flights. "
                f"Fully crewed: {covered}. "
                f"Partial/cancel: {len(affected_flights) - covered}. "
                f"FAR 117 checks: {'enabled' if legality_available else 'fallback mode'}."
            ),
        }

        log.info(
            f"[CrewAgent] {covered}/{len(affected_flights)} flights fully crewed "
            f"| FAR117={'on' if legality_available else 'off'} "
            f"| cost={total_cost:.2f}"
        )
        return proposal