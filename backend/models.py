from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey,
    Enum, Text, JSON, Index
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

Base = declarative_base()

def gen_uuid():
    return str(uuid.uuid4())


# ── Enums ──────────────────────────────────────────────────────────────────────

class FlightStatus(str, enum.Enum):
    SCHEDULED   = "scheduled"
    DEPARTED    = "departed"
    ARRIVED     = "arrived"
    DELAYED     = "delayed"
    CANCELLED   = "cancelled"
    DIVERTED    = "diverted"

class DisruptionType(str, enum.Enum):
    WEATHER         = "weather"
    MECHANICAL      = "mechanical"
    CREW            = "crew"
    ATC             = "atc"
    AIRPORT         = "airport"
    UNKNOWN         = "unknown"

class DisruptionSeverity(str, enum.Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"

class CrewRole(str, enum.Enum):
    CAPTAIN         = "captain"
    FIRST_OFFICER   = "first_officer"
    FLIGHT_ATTENDANT = "flight_attendant"

class CrewStatus(str, enum.Enum):
    AVAILABLE  = "available"
    ON_DUTY    = "on_duty"
    REST       = "rest"
    STANDBY    = "standby"
    OFF        = "off"

class AircraftStatus(str, enum.Enum):
    AVAILABLE      = "available"
    IN_FLIGHT      = "in_flight"
    MAINTENANCE    = "maintenance"
    AOG            = "aog"           # Aircraft on ground (unscheduled)

class RecoveryActionType(str, enum.Enum):
    AIRCRAFT_SWAP       = "aircraft_swap"
    CREW_REASSIGN       = "crew_reassign"
    FLIGHT_DELAY        = "flight_delay"
    FLIGHT_CANCEL       = "flight_cancel"
    PASSENGER_REBOOK    = "passenger_rebook"
    CREW_REPOSITION     = "crew_reposition"

class FareClass(str, enum.Enum):
    FIRST    = "first"
    BUSINESS = "business"
    ECONOMY  = "economy"


# ── Airport & Network ──────────────────────────────────────────────────────────

class Airport(Base):
    __tablename__ = "airports"

    id          = Column(String(10), primary_key=True)   # IATA code e.g. "JFK"
    icao        = Column(String(10), nullable=True)
    name        = Column(String(200), nullable=False)
    city        = Column(String(100))
    country     = Column(String(100))
    latitude    = Column(Float, nullable=False)
    longitude   = Column(Float, nullable=False)
    altitude_ft = Column(Integer, default=0)
    timezone    = Column(String(50))
    is_hub      = Column(Boolean, default=False)

    # relationships
    departures  = relationship("Flight", foreign_keys="Flight.origin_id",      back_populates="origin")
    arrivals    = relationship("Flight", foreign_keys="Flight.destination_id",  back_populates="destination")
    crew_bases  = relationship("CrewMember", back_populates="base_airport")
    aircraft_stationed = relationship("Aircraft", back_populates="home_airport")


class Route(Base):
    __tablename__ = "routes"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    origin_id       = Column(String(10), ForeignKey("airports.id"), nullable=False)
    destination_id  = Column(String(10), ForeignKey("airports.id"), nullable=False)
    airline_code    = Column(String(5))
    distance_km     = Column(Float)
    avg_duration_min = Column(Integer)
    is_active       = Column(Boolean, default=True)

    origin      = relationship("Airport", foreign_keys=[origin_id])
    destination = relationship("Airport", foreign_keys=[destination_id])

    __table_args__ = (
        Index("ix_routes_origin_dest", "origin_id", "destination_id"),
    )


# ── Flights ────────────────────────────────────────────────────────────────────

class Flight(Base):
    __tablename__ = "flights"

    id                  = Column(String(36), primary_key=True, default=gen_uuid)
    flight_number       = Column(String(10), nullable=False)    # e.g. "AA302"
    airline_code        = Column(String(5),  nullable=False)
    origin_id           = Column(String(10), ForeignKey("airports.id"), nullable=False)
    destination_id      = Column(String(10), ForeignKey("airports.id"), nullable=False)
    aircraft_id         = Column(String(36), ForeignKey("aircraft.id"), nullable=True)

    scheduled_departure = Column(DateTime, nullable=False)
    scheduled_arrival   = Column(DateTime, nullable=False)
    actual_departure    = Column(DateTime, nullable=True)
    actual_arrival      = Column(DateTime, nullable=True)

    status              = Column(Enum(FlightStatus), default=FlightStatus.SCHEDULED)
    delay_minutes       = Column(Integer, default=0)
    cancellation_reason = Column(String(200), nullable=True)
    divert_airport_id   = Column(String(10), ForeignKey("airports.id"), nullable=True)

    tail_number         = Column(String(20), nullable=True)
    capacity            = Column(Integer, default=150)
    booked_seats        = Column(Integer, default=0)

    # BTS metadata (for historical data)
    bts_carrier         = Column(String(5),  nullable=True)
    bts_origin          = Column(String(5),  nullable=True)
    bts_dest            = Column(String(5),  nullable=True)
    year                = Column(Integer,    nullable=True)
    month               = Column(Integer,    nullable=True)
    day_of_week         = Column(Integer,    nullable=True)

    # relationships
    origin              = relationship("Airport", foreign_keys=[origin_id],      back_populates="departures")
    destination         = relationship("Airport", foreign_keys=[destination_id], back_populates="arrivals")
    aircraft            = relationship("Aircraft", back_populates="flights")
    crew_assignments    = relationship("CrewAssignment", back_populates="flight")
    passenger_itins     = relationship("PassengerItinerary", back_populates="flight")
    disruption_events   = relationship("DisruptionEvent",
                                        foreign_keys="DisruptionEvent.root_flight_id",
                                        back_populates="root_flight")

    __table_args__ = (
        Index("ix_flights_flight_number", "flight_number"),
        Index("ix_flights_scheduled_dep", "scheduled_departure"),
        Index("ix_flights_status", "status"),
    )


# ── Disruptions ────────────────────────────────────────────────────────────────

class DisruptionEvent(Base):
    __tablename__ = "disruption_events"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    root_flight_id  = Column(String(36), ForeignKey("flights.id"), nullable=False)
    detected_at     = Column(DateTime,   nullable=False)
    disruption_type = Column(Enum(DisruptionType), default=DisruptionType.UNKNOWN)
    severity        = Column(Enum(DisruptionSeverity), default=DisruptionSeverity.LOW)

    # serialised list of affected flight IDs + risk scores
    # e.g. [{"flight_id": "...", "risk_score": 0.87, "delay_estimate_min": 120}]
    affected_flights_json   = Column(JSON, default=list)

    total_affected_flights  = Column(Integer, default=0)
    total_affected_pax      = Column(Integer, default=0)
    estimated_delay_min     = Column(Integer, default=0)

    weather_metar           = Column(Text, nullable=True)
    weather_condition       = Column(String(100), nullable=True)

    resolved                = Column(Boolean, default=False)
    resolved_at             = Column(DateTime, nullable=True)

    recovery_plan_id        = Column(String(36), ForeignKey("recovery_plans.id"), nullable=True)

    root_flight     = relationship("Flight", foreign_keys=[root_flight_id], back_populates="disruption_events")
    recovery_plan   = relationship("RecoveryPlan", back_populates="disruption_event")

    __table_args__ = (
        Index("ix_disruption_detected_at", "detected_at"),
        Index("ix_disruption_severity", "severity"),
    )


# ── Recovery ───────────────────────────────────────────────────────────────────

class RecoveryPlan(Base):
    __tablename__ = "recovery_plans"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    created_at      = Column(DateTime, nullable=False)
    finalized       = Column(Boolean, default=False)

    # per-agent raw proposals (JSON blobs)
    fleet_proposal_json     = Column(JSON, nullable=True)
    crew_proposal_json      = Column(JSON, nullable=True)
    passenger_proposal_json = Column(JSON, nullable=True)

    # final merged plan
    final_actions_json      = Column(JSON, default=list)

    # scoring
    cancellations_avoided   = Column(Integer, default=0)
    misconnects_avoided     = Column(Integer, default=0)
    total_delay_reduction   = Column(Integer, default=0)   # minutes

    # LLM-generated summary
    advisory_text           = Column(Text, nullable=True)

    disruption_event        = relationship("DisruptionEvent", back_populates="recovery_plan")
    actions                 = relationship("RecoveryAction",  back_populates="recovery_plan")


class RecoveryAction(Base):
    __tablename__ = "recovery_actions"

    id                  = Column(String(36), primary_key=True, default=gen_uuid)
    recovery_plan_id    = Column(String(36), ForeignKey("recovery_plans.id"), nullable=False)
    action_type         = Column(Enum(RecoveryActionType), nullable=False)
    agent_source        = Column(String(20))   # "fleet" | "crew" | "passenger"

    flight_id           = Column(String(36), ForeignKey("flights.id"), nullable=True)
    crew_member_id      = Column(String(36), ForeignKey("crew_members.id"), nullable=True)
    aircraft_id         = Column(String(36), ForeignKey("aircraft.id"), nullable=True)
    passenger_id        = Column(String(36), ForeignKey("passengers.id"), nullable=True)

    description         = Column(Text)
    cost_score          = Column(Float, default=0.0)
    impact_score        = Column(Float, default=0.0)
    feasible            = Column(Boolean, default=True)
    conflict_flag       = Column(Boolean, default=False)
    conflict_reason     = Column(String(200), nullable=True)

    metadata_json       = Column(JSON, nullable=True)

    recovery_plan       = relationship("RecoveryPlan", back_populates="actions")


# ── Crew ───────────────────────────────────────────────────────────────────────

class CrewMember(Base):
    __tablename__ = "crew_members"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    employee_id     = Column(String(20), unique=True, nullable=False)
    name            = Column(String(100), nullable=False)
    role            = Column(Enum(CrewRole), nullable=False)
    base_airport_id = Column(String(10), ForeignKey("airports.id"))
    status          = Column(Enum(CrewStatus), default=CrewStatus.AVAILABLE)

    # duty time tracking (hours)
    max_duty_hours          = Column(Float, default=14.0)
    current_duty_hours      = Column(Float, default=0.0)
    min_rest_hours          = Column(Float, default=10.0)
    last_rest_start         = Column(DateTime, nullable=True)
    cumulative_duty_7day    = Column(Float, default=0.0)   # FAR 117 rolling window

    qualifications_json     = Column(JSON, default=list)   # ["B737", "A320"]
    current_location_id     = Column(String(10), ForeignKey("airports.id"), nullable=True)

    base_airport    = relationship("Airport", foreign_keys=[base_airport_id], back_populates="crew_bases")
    current_location = relationship("Airport", foreign_keys=[current_location_id])
    assignments     = relationship("CrewAssignment", back_populates="crew_member")


class CrewAssignment(Base):
    __tablename__ = "crew_assignments"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    flight_id       = Column(String(36), ForeignKey("flights.id"),      nullable=False)
    crew_member_id  = Column(String(36), ForeignKey("crew_members.id"), nullable=False)
    role_on_flight  = Column(Enum(CrewRole))
    is_active       = Column(Boolean, default=True)    # False = reassigned away
    assigned_at     = Column(DateTime, nullable=True)

    flight      = relationship("Flight",     back_populates="crew_assignments")
    crew_member = relationship("CrewMember", back_populates="assignments")


# ── Aircraft ───────────────────────────────────────────────────────────────────

class Aircraft(Base):
    __tablename__ = "aircraft"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    tail_number     = Column(String(20), unique=True, nullable=False)
    aircraft_type   = Column(String(20), nullable=False)   # "B737", "A320"
    capacity        = Column(Integer, nullable=False)
    range_km        = Column(Integer, default=5000)
    status          = Column(Enum(AircraftStatus), default=AircraftStatus.AVAILABLE)
    home_airport_id = Column(String(10), ForeignKey("airports.id"))
    current_airport_id = Column(String(10), ForeignKey("airports.id"), nullable=True)

    # maintenance windows: [{"start": ISO, "end": ISO, "type": "scheduled|AOG"}]
    maintenance_windows_json = Column(JSON, default=list)

    home_airport    = relationship("Airport", foreign_keys=[home_airport_id], back_populates="aircraft_stationed")
    current_airport = relationship("Airport", foreign_keys=[current_airport_id])
    flights         = relationship("Flight", back_populates="aircraft")


# ── Passengers ────────────────────────────────────────────────────────────────

class Passenger(Base):
    __tablename__ = "passengers"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    pnr             = Column(String(10), unique=True, nullable=False)   # booking reference
    name            = Column(String(100))
    fare_class      = Column(Enum(FareClass), default=FareClass.ECONOMY)
    frequent_flyer  = Column(Boolean, default=False)
    contact_email   = Column(String(200), nullable=True)

    itineraries     = relationship("PassengerItinerary", back_populates="passenger")


class PassengerItinerary(Base):
    __tablename__ = "passenger_itineraries"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    passenger_id    = Column(String(36), ForeignKey("passengers.id"), nullable=False)
    flight_id       = Column(String(36), ForeignKey("flights.id"),    nullable=False)

    # leg position in the journey: 1 = first leg, 2 = connection, etc.
    leg_number          = Column(Integer, default=1)
    connection_time_min = Column(Integer, nullable=True)   # to next leg
    misconnect_risk     = Column(Float, default=0.0)       # 0–1 probability
    rebooked            = Column(Boolean, default=False)
    rebooked_flight_id  = Column(String(36), ForeignKey("flights.id"), nullable=True)

    passenger   = relationship("Passenger", back_populates="itineraries")
    flight      = relationship("Flight", foreign_keys=[flight_id], back_populates="passenger_itins")
    rebooked_flight = relationship("Flight", foreign_keys=[rebooked_flight_id])

    __table_args__ = (
        Index("ix_pax_itin_flight", "flight_id"),
        Index("ix_pax_itin_misconnect", "misconnect_risk"),
    )


# ── Weather Snapshots ─────────────────────────────────────────────────────────

class WeatherSnapshot(Base):
    __tablename__ = "weather_snapshots"

    id              = Column(String(36), primary_key=True, default=gen_uuid)
    airport_id      = Column(String(10), ForeignKey("airports.id"), nullable=False)
    observed_at     = Column(DateTime, nullable=False)
    raw_metar       = Column(Text, nullable=True)

    visibility_sm   = Column(Float, nullable=True)
    wind_speed_kt   = Column(Integer, nullable=True)
    wind_gust_kt    = Column(Integer, nullable=True)
    ceiling_ft      = Column(Integer, nullable=True)    # cloud base
    temperature_c   = Column(Float, nullable=True)
    condition_code  = Column(String(20), nullable=True) # "VFR","IFR","LIFR","MVFR"

    airport         = relationship("Airport")

    __table_args__ = (
        Index("ix_weather_airport_time", "airport_id", "observed_at"),
    )