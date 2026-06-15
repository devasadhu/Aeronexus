"""
agents/crew_legality.py

FAR 117 Crew Legality Engine — used by CrewAgent to validate assignments.

Implements (simplified but defensible) FAR Part 117:
  - Max FDP by report time and number of flight segments (Table B)
  - Minimum rest: 10h with 8h sleep opportunity (or 8h reduced rest)
  - 7-day cumulative: 60h flight duty
  - 28-day cumulative: 100h flight duty
  - 365-day cumulative: 1000h flight time
  - Calendar-day FDP extension rules (not implemented — flagged as TODO)

All times in minutes internally. Input datetimes as ISO strings or datetime objects.

Usage:
    from agents.crew_legality import check_legality, LegalityVerdict

    verdict = check_legality(crew, flight, aircraft_type)
    if not verdict.legal:
        print(verdict.violations)   # list of violation codes + messages
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

# ── FAR 117 Table B: Max FDP (minutes) by report_local_hour × segment_count ──
# Rows: report hour (0–23). Cols: number of flight segments (1, 2, 3, 4, 5+).
# Source: FAR 117.13, Table B (unaugmented, lineholder)
_FDP_TABLE_MINUTES: dict[int, list[int]] = {
    #  hr   1seg  2seg  3seg  4seg  5+seg
     0: [9*60,    9*60,    9*60,    9*60,    9*60],
     1: [9*60,    9*60,    9*60,    9*60,    9*60],
     2: [9*60,    9*60,    9*60,    9*60,    9*60],
     3: [9*60,    9*60,    9*60,    9*60,    9*60],
     4: [9*60,    9*60,    9*60,    9*60,    9*60],
     5: [9*60,    9*60,    9*60,    9*60,    9*60],
     6: [13*60,   13*60,   12*60,   12*60,   11*60+30],
     7: [14*60,   14*60,   13*60,   13*60,   12*60+30],
     8: [14*60,   14*60,   13*60,   13*60,   12*60+30],
     9: [14*60,   14*60,   13*60,   13*60,   12*60+30],
    10: [14*60,   14*60,   13*60,   13*60,   12*60+30],
    11: [14*60,   14*60,   13*60,   13*60,   12*60+30],
    12: [14*60,   14*60,   13*60,   13*60,   12*60+30],
    13: [13*60,   13*60,   12*60,   12*60,   11*60+30],
    14: [13*60,   13*60,   12*60,   12*60,   11*60+30],
    15: [13*60,   13*60,   12*60,   12*60,   11*60+30],
    16: [12*60,   12*60,   11*60+30, 11*60+30, 11*60],
    17: [12*60,   12*60,   11*60+30, 11*60+30, 11*60],
    18: [12*60,   12*60,   11*60+30, 11*60+30, 11*60],
    19: [12*60,   12*60,   11*60+30, 11*60+30, 11*60],
    20: [11*60+30, 11*60+30, 11*60,  11*60,   10*60+30],
    21: [11*60,   11*60,   10*60+30, 10*60+30, 10*60],
    22: [10*60,   10*60,   10*60,   10*60,    9*60+30],
    23: [ 9*60,    9*60,    9*60,    9*60,    9*60],
}

# Cumulative limits (hours → minutes)
_LIMIT_7DAY_MIN   = 60  * 60    # 60h
_LIMIT_28DAY_MIN  = 100 * 60    # 100h
_LIMIT_365DAY_MIN = 1000 * 60   # 1000h flight time

# Rest minimums
_MIN_REST_MIN         = 10 * 60   # 10h between duties
_MIN_SLEEP_OPP_MIN    =  8 * 60   # 8h sleep opportunity within rest
_REDUCED_REST_MIN     =  8 * 60   # reduced rest floor (with airline program)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class LegalityVerdict:
    legal: bool = True
    violations: list[str] = field(default_factory=list)
    warnings: list[str]   = field(default_factory=list)
    max_fdp_min: int       = 0
    used_fdp_min: float    = 0.0
    remaining_duty_min: float = 0.0

    def add_violation(self, code: str, msg: str):
        self.legal = False
        self.violations.append(f"[{code}] {msg}")

    def add_warning(self, msg: str):
        self.warnings.append(f"[WARN] {msg}")

    def summary(self) -> str:
        if self.legal:
            extra = f" | warnings: {len(self.warnings)}" if self.warnings else ""
            return f"LEGAL (FDP used {self.used_fdp_min:.0f}/{self.max_fdp_min}min{extra})"
        return "ILLEGAL — " + "; ".join(self.violations)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(s) -> Optional[datetime]:
    if isinstance(s, datetime):
        return s
    if s:
        try:
            return datetime.fromisoformat(str(s))
        except ValueError:
            pass
    return None


def _fdp_limit_minutes(report_hour: int, num_segments: int) -> int:
    """Look up FAR 117 Table B."""
    report_hour = max(0, min(23, int(report_hour)))
    seg_idx     = min(num_segments, 5) - 1   # 0-indexed, cap at 5
    return _FDP_TABLE_MINUTES[report_hour][seg_idx]


def _flight_duration_minutes(dep, arr) -> float:
    dep_dt = _parse_dt(dep)
    arr_dt = _parse_dt(arr)
    if dep_dt and arr_dt:
        if arr_dt < dep_dt:
            arr_dt += timedelta(days=1)
        return (arr_dt - dep_dt).total_seconds() / 60
    return 180.0   # default 3h


def _rest_since_last_duty(crew: dict, report_dt: datetime) -> Optional[float]:
    """
    Returns rest period in minutes since last duty ended, or None if unknown.
    Uses last_duty_end_utc from crew record.
    """
    last_end = _parse_dt(crew.get("last_duty_end_utc"))
    if last_end is None:
        return None
    return (report_dt - last_end).total_seconds() / 60


# ── Main legality check ───────────────────────────────────────────────────────

def check_legality(
    crew:          dict,
    flight:        dict,
    aircraft_type: str  = "",
    num_segments:  int  = 1,
    report_offset_min: int = 60,   # crew reports this many minutes before departure
) -> LegalityVerdict:
    """
    Full FAR 117 legality check for one crew member on one flight.

    crew dict expected fields (all optional — missing = most permissive assumed):
      status                  "available" | "standby" | "rest" | "off"
      qualifications_json     list of aircraft type strings
      current_location_id     IATA airport code
      current_duty_hours      float — hours already on duty today
      cumulative_duty_7day    float — hours in last 7 days
      cumulative_duty_28day   float — hours in last 28 days
      cumulative_flight_365day float — flight hours in last 365 days
      last_duty_end_utc       ISO datetime string — when last duty period ended
      max_duty_hours          float — individual override (default 14h)

    flight dict expected fields:
      origin_id               IATA
      scheduled_departure     ISO datetime
      scheduled_arrival       ISO datetime
      capacity                int
    """
    verdict = LegalityVerdict()

    # ── 1. Status check ───────────────────────────────────────────────────────
    status = crew.get("status", "available")
    if status in ("rest", "off"):
        verdict.add_violation(
            "STATUS",
            f"Crew status is '{status}' — not available for duty"
        )
        return verdict   # no point checking further

    # ── 2. Qualification check ────────────────────────────────────────────────
    if aircraft_type:
        quals = crew.get("qualifications_json", [])
        if isinstance(quals, str):
            import json as _json
            try:
                quals = _json.loads(quals)
            except Exception:
                quals = [quals]
        if aircraft_type not in quals:
            verdict.add_violation(
                "QUAL",
                f"Not type-rated on {aircraft_type} (qualified: {quals or 'none listed'})"
            )

    # ── 3. Rest requirement ───────────────────────────────────────────────────
    dep_dt = _parse_dt(flight.get("scheduled_departure"))
    if dep_dt:
        report_dt  = dep_dt - timedelta(minutes=report_offset_min)
        rest_min   = _rest_since_last_duty(crew, report_dt)

        if rest_min is not None:
            if rest_min < _REDUCED_REST_MIN:
                verdict.add_violation(
                    "REST_MIN",
                    f"Insufficient rest: {rest_min:.0f}min < {_REDUCED_REST_MIN}min minimum"
                )
            elif rest_min < _MIN_REST_MIN:
                verdict.add_warning(
                    f"Reduced rest: {rest_min:.0f}min (standard is {_MIN_REST_MIN}min) "
                    f"— requires airline reduced-rest program"
                )
        else:
            verdict.add_warning("last_duty_end_utc not set — rest period unverifiable")
    else:
        dep_dt    = datetime.utcnow()
        report_dt = dep_dt - timedelta(minutes=report_offset_min)

    # ── 4. FDP limit (Table B) ────────────────────────────────────────────────
    report_hour     = report_dt.hour
    max_fdp_min     = _fdp_limit_minutes(report_hour, num_segments)
    flight_dur_min  = _flight_duration_minutes(
        flight.get("scheduled_departure"),
        flight.get("scheduled_arrival")
    )

    # current duty already accumulated today
    current_duty_min = crew.get("current_duty_hours", 0.0) * 60

    # FDP = time from report to block-in (current duty + pre-departure prep + flight)
    pre_dep_buffer  = report_offset_min   # already at airport from report time
    used_fdp_min    = current_duty_min + pre_dep_buffer + flight_dur_min

    verdict.max_fdp_min   = max_fdp_min
    verdict.used_fdp_min  = used_fdp_min
    verdict.remaining_duty_min = max(0, max_fdp_min - used_fdp_min)

    # also enforce individual max_duty_hours override
    individual_max_min = crew.get("max_duty_hours", 14.0) * 60
    effective_max_min  = min(max_fdp_min, individual_max_min)

    if used_fdp_min > effective_max_min:
        verdict.add_violation(
            "FDP",
            f"FDP would be {used_fdp_min:.0f}min, limit is {effective_max_min:.0f}min "
            f"(report {report_hour:02d}:xx, {num_segments} segment(s))"
        )
    elif used_fdp_min > effective_max_min * 0.90:
        verdict.add_warning(
            f"FDP at {used_fdp_min/effective_max_min*100:.0f}% of limit "
            f"({used_fdp_min:.0f}/{effective_max_min:.0f}min)"
        )

    # ── 5. 7-day cumulative ───────────────────────────────────────────────────
    cum_7day_min  = crew.get("cumulative_duty_7day",  0.0) * 60
    proj_7day_min = cum_7day_min + flight_dur_min
    if proj_7day_min > _LIMIT_7DAY_MIN:
        verdict.add_violation(
            "CUM_7D",
            f"7-day total would be {proj_7day_min/60:.1f}h, limit is {_LIMIT_7DAY_MIN/60:.0f}h"
        )
    elif proj_7day_min > _LIMIT_7DAY_MIN * 0.90:
        verdict.add_warning(
            f"7-day duty at {proj_7day_min/60:.1f}h (limit {_LIMIT_7DAY_MIN/60:.0f}h)"
        )

    # ── 6. 28-day cumulative ──────────────────────────────────────────────────
    cum_28day_min  = crew.get("cumulative_duty_28day", 0.0) * 60
    proj_28day_min = cum_28day_min + flight_dur_min
    if proj_28day_min > _LIMIT_28DAY_MIN:
        verdict.add_violation(
            "CUM_28D",
            f"28-day total would be {proj_28day_min/60:.1f}h, limit is {_LIMIT_28DAY_MIN/60:.0f}h"
        )

    # ── 7. 365-day flight time ────────────────────────────────────────────────
    cum_365_min  = crew.get("cumulative_flight_365day", 0.0) * 60
    proj_365_min = cum_365_min + flight_dur_min
    if proj_365_min > _LIMIT_365DAY_MIN:
        verdict.add_violation(
            "CUM_365D",
            f"365-day flight time would be {proj_365_min/60:.0f}h, "
            f"limit is {_LIMIT_365DAY_MIN/60:.0f}h"
        )

    log.debug(
        f"[Legality] {crew.get('name','?')} | {crew.get('role','?')} | "
        f"flight {flight.get('flight_number','?')} | {verdict.summary()}"
    )
    return verdict


# ── Batch helper used by CrewAgent ────────────────────────────────────────────

def filter_legal_crew(
    candidates:    list[dict],
    flight:        dict,
    aircraft_type: str = "",
    num_segments:  int = 1,
) -> list[tuple[dict, LegalityVerdict]]:
    """
    Filter a list of crew to those legally eligible for a flight.
    Returns list of (crew_dict, verdict) for legal crew only,
    sorted by remaining FDP (most margin first).
    """
    legal = []
    for c in candidates:
        v = check_legality(c, flight, aircraft_type, num_segments)
        if v.legal:
            legal.append((c, v))

    legal.sort(key=lambda x: -x[1].remaining_duty_min)
    return legal


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    from pathlib import Path
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    crew_list = json.loads(Path("data/processed/crew.json").read_text())
    flights   = json.loads(Path("data/processed/flights.json").read_text())

    flight = flights[0]
    log.info(f"Checking crew for flight {flight.get('flight_number')} "
             f"({flight.get('origin_id')}→{flight.get('destination_id')})")

    for crew in crew_list[:10]:
        v = check_legality(crew, flight, aircraft_type="B737")
        log.info(f"  {crew['name']:30s} {crew['role']:18s} → {v.summary()}")