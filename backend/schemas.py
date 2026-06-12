from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from .models import (
    FlightStatus, DisruptionType, DisruptionSeverity,
    CrewRole, CrewStatus, AircraftStatus, RecoveryActionType, FareClass
)


# ── Shared ─────────────────────────────────────────────────────────────────────

class AffectedFlight(BaseModel):
    flight_id:          str
    flight_number:      str
    risk_score:         float   = Field(..., ge=0.0, le=1.0)
    delay_estimate_min: int     = 0
    reason:             str     = ""


# ── Airport ────────────────────────────────────────────────────────────────────

class AirportOut(BaseModel):
    id:         str
    name:       str
    city:       Optional[str]
    country:    Optional[str]
    latitude:   float
    longitude:  float
    is_hub:     bool

    class Config:
        from_attributes = True


# ── Flight ─────────────────────────────────────────────────────────────────────

class FlightOut(BaseModel):
    id:                  str
    flight_number:       str
    airline_code:        str
    origin_id:           str
    destination_id:      str
    scheduled_departure: datetime
    scheduled_arrival:   datetime
    actual_departure:    Optional[datetime]
    actual_arrival:      Optional[datetime]
    status:              FlightStatus
    delay_minutes:       int
    tail_number:         Optional[str]
    capacity:            int
    booked_seats:        int

    class Config:
        from_attributes = True


class FlightUpdate(BaseModel):
    status:           Optional[FlightStatus] = None
    delay_minutes:    Optional[int]          = None
    actual_departure: Optional[datetime]     = None
    actual_arrival:   Optional[datetime]     = None


# ── Disruption ─────────────────────────────────────────────────────────────────

class DisruptionEventOut(BaseModel):
    id:                     str
    root_flight_id:         str
    detected_at:            datetime
    disruption_type:        DisruptionType
    severity:               DisruptionSeverity
    affected_flights_json:  List[AffectedFlight]
    total_affected_flights: int
    total_affected_pax:     int
    estimated_delay_min:    int
    weather_condition:      Optional[str]
    resolved:               bool

    class Config:
        from_attributes = True


class DisruptionCreate(BaseModel):
    root_flight_id:  str
    disruption_type: DisruptionType = DisruptionType.UNKNOWN
    delay_minutes:   int            = 0
    weather_metar:   Optional[str]  = None


# ── Crew ───────────────────────────────────────────────────────────────────────

class CrewMemberOut(BaseModel):
    id:                      str
    employee_id:             str
    name:                    str
    role:                    CrewRole
    base_airport_id:         str
    status:                  CrewStatus
    current_duty_hours:      float
    max_duty_hours:          float
    qualifications_json:     List[str]
    current_location_id:     Optional[str]

    class Config:
        from_attributes = True


# ── Aircraft ───────────────────────────────────────────────────────────────────

class AircraftOut(BaseModel):
    id:                 str
    tail_number:        str
    aircraft_type:      str
    capacity:           int
    status:             AircraftStatus
    home_airport_id:    str
    current_airport_id: Optional[str]

    class Config:
        from_attributes = True


# ── Recovery ───────────────────────────────────────────────────────────────────

class RecoveryActionOut(BaseModel):
    id:             str
    action_type:    RecoveryActionType
    agent_source:   str
    flight_id:      Optional[str]
    description:    str
    cost_score:     float
    impact_score:   float
    feasible:       bool
    conflict_flag:  bool

    class Config:
        from_attributes = True


class RecoveryPlanOut(BaseModel):
    id:                     str
    created_at:             datetime
    finalized:              bool
    cancellations_avoided:  int
    misconnects_avoided:    int
    total_delay_reduction:  int
    advisory_text:          Optional[str]
    actions:                List[RecoveryActionOut] = []

    class Config:
        from_attributes = True


# ── Agent I/O (used internally between agents and coordinator) ─────────────────

class AgentProposal(BaseModel):
    agent:      str          # "fleet" | "crew" | "passenger"
    actions:    List[Dict[str, Any]]
    cost:       float = 0.0
    confidence: float = 1.0
    notes:      str   = ""


class RecoveryContext(BaseModel):
    disruption:         DisruptionEventOut
    affected_flights:   List[FlightOut]
    available_crew:     List[CrewMemberOut]
    available_aircraft: List[AircraftOut]
    constraints:        Dict[str, Any] = {}


# ── Advisory ───────────────────────────────────────────────────────────────────

class AdvisoryRequest(BaseModel):
    disruption_id:   str
    recovery_plan_id: str

class AdvisoryResponse(BaseModel):
    summary:    str
    bullets:    List[str]
    severity:   DisruptionSeverity
    generated_at: datetime