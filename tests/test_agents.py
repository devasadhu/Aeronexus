"""
tests/test_agents.py

Tests for FleetAgent, CrewAgent, PassengerAgent, CoordinatorAgent.
Uses in-memory synthetic data — no DB, no external services needed.
"""

import sys, json, pytest
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_flight(flight_id="f1", fn="AA101", origin="ATL", dest="ORD",
                  status="delayed", delay=90, capacity=150, booked=120,
                  dep_offset_h=2, duration_h=2):
    dep = datetime.utcnow() + timedelta(hours=dep_offset_h)
    arr = dep + timedelta(hours=duration_h)
    return {
        "id": flight_id, "flight_number": fn,
        "origin_id": origin, "destination_id": dest,
        "status": status, "delay_minutes": delay,
        "capacity": capacity, "booked_seats": booked,
        "scheduled_departure": dep.isoformat(),
        "scheduled_arrival":   arr.isoformat(),
    }


def _make_aircraft(ac_id="ac1", tail="N-ABC1", atype="B737", cap=160,
                    status="available", home="ATL", current="ATL", maint=None):
    return {
        "id": ac_id, "tail_number": tail, "aircraft_type": atype,
        "capacity": cap, "status": status,
        "home_airport_id": home, "current_airport_id": current,
        "maintenance_windows_json": maint or [],
    }


def _make_crew(emp_id="CPT001", name="Alice Chen", role="captain",
               base="ATL", status="available", duty=3.0, max_duty=14.0,
               cum_7day=20.0, quals=None, location="ATL"):
    return {
        "id": emp_id, "employee_id": emp_id, "name": name,
        "role": role, "base_airport_id": base,
        "status": status, "max_duty_hours": max_duty,
        "current_duty_hours": duty, "min_rest_hours": 10.0,
        "cumulative_duty_7day": cum_7day,
        "qualifications_json": quals or ["B737", "A320"],
        "current_location_id": location,
    }


def _make_passenger(pax_id="p1", pnr="ABC123", fare="economy", ff=False):
    return {"id": pax_id, "pnr": pnr, "name": "Test User",
             "fare_class": fare, "frequent_flyer": ff}


def _make_itin(itin_id="i1", pax_id="p1", flight_id="f1",
                leg=1, conn_time=None, risk=0.0):
    return {"id": itin_id, "passenger_id": pax_id, "flight_id": flight_id,
             "leg_number": leg, "connection_time_min": conn_time,
             "misconnect_risk": risk, "rebooked": False}


def _make_disruption(flight_id="f1", dtype="weather", delay=90):
    return {
        "id": "evt1", "root_flight_id": flight_id,
        "disruption_type": dtype, "severity": "high",
        "severity_score": 60.0, "severity_breakdown": {},
        "affected_flights_json": [],
        "total_affected_flights": 1, "total_affected_pax": 120,
        "estimated_delay_min": delay,
        "weather_metar": None, "weather_condition": None,
        "resolved": False, "resolved_at": None, "recovery_plan_id": None,
    }


# ── FleetAgent tests ──────────────────────────────────────────────────────────

class TestFleetAgent:
    from agents.fleet_agent import FleetAgent

    def _run(self, flights, aircraft):
        from agents.fleet_agent import FleetAgent
        ctx = {
            "disruption": _make_disruption(),
            "affected_flights": flights,
            "available_aircraft": aircraft,
            "original_aircraft": {},
        }
        return FleetAgent().run(ctx)

    def test_swap_proposed_when_aircraft_available(self):
        flight = _make_flight(origin="ATL")
        ac     = _make_aircraft(current="ATL", status="available")
        result = self._run([flight], [ac])
        swaps  = [a for a in result["actions"] if a["action_type"] == "aircraft_swap"]
        assert len(swaps) == 1

    def test_cancel_when_no_aircraft(self):
        flight = _make_flight(origin="ATL")
        result = self._run([flight], [])
        cancels = [a for a in result["actions"] if a["action_type"] == "flight_cancel"]
        assert len(cancels) == 1

    def test_no_maintenance_overlap(self):
        dep = datetime.utcnow() + timedelta(hours=2)
        arr = dep + timedelta(hours=3)
        maint = [{"start": dep.isoformat(), "end": arr.isoformat(), "type": "scheduled"}]
        flight = _make_flight(origin="ATL")
        ac_bad  = _make_aircraft(ac_id="ac_bad", tail="N-BAD1", current="ATL", maint=maint)
        ac_good = _make_aircraft(ac_id="ac_ok",  tail="N-GD01", current="ATL")
        result  = self._run([flight], [ac_bad, ac_good])
        swaps   = [a for a in result["actions"] if a["action_type"] == "aircraft_swap"]
        assert swaps[0]["metadata"]["new_tail"] == "N-GD01"

    def test_no_double_assignment(self):
        f1 = _make_flight(flight_id="f1", fn="AA101", origin="ATL")
        f2 = _make_flight(flight_id="f2", fn="AA102", origin="ATL")
        ac = _make_aircraft(current="ATL")   # only one aircraft
        result = self._run([f1, f2], [ac])
        swaps   = [a for a in result["actions"] if a["action_type"] == "aircraft_swap"]
        cancels = [a for a in result["actions"] if a["action_type"] == "flight_cancel"]
        assert len(swaps) == 1
        assert len(cancels) == 1

    def test_capacity_downgrade_penalised(self):
        from agents.fleet_agent import _score_candidate
        orig = _make_aircraft(cap=200)
        small = _make_aircraft(ac_id="ac2", tail="N-SM01", cap=100)
        large = _make_aircraft(ac_id="ac3", tail="N-LG01", cap=200)
        score_small = _score_candidate(small, {}, orig)
        score_large = _score_candidate(large, {}, orig)
        assert score_small > score_large   # larger is better (lower score)

    def test_proposal_has_cost_and_notes(self):
        flight = _make_flight(origin="ATL")
        ac     = _make_aircraft(current="ATL")
        result = self._run([flight], [ac])
        assert "cost" in result
        assert "notes" in result
        assert result["agent"] == "fleet"

    def test_reposition_flagged(self):
        flight = _make_flight(origin="ATL")
        ac     = _make_aircraft(current="ORD")  # different airport
        result = self._run([flight], [ac])
        swaps  = [a for a in result["actions"] if a["action_type"] == "aircraft_swap"]
        if swaps:
            assert swaps[0]["metadata"]["needs_reposition"] is True


# ── CrewAgent tests ───────────────────────────────────────────────────────────

class TestCrewAgent:

    def _run(self, flights, crew, type_map=None):
        from agents.crew_agent import CrewAgent
        ctx = {
            "disruption": _make_disruption(),
            "affected_flights": flights,
            "available_crew": crew,
            "aircraft_type_map": type_map or {},
        }
        return CrewAgent().run(ctx)

    def _full_crew(self, location="ATL", quals=None):
        q = quals or ["B737", "A320"]
        return [
            _make_crew("CPT001", "Alice",  "captain",          location=location, quals=q),
            _make_crew("FO001",  "Bob",    "first_officer",    location=location, quals=q),
            _make_crew("FA001",  "Carol",  "flight_attendant", location=location, quals=q),
            _make_crew("FA002",  "Dave",   "flight_attendant", location=location, quals=q),
            _make_crew("FA003",  "Eve",    "flight_attendant", location=location, quals=q),
        ]

    def test_fully_crewed(self):
        flight = _make_flight()
        result = self._run([flight], self._full_crew())
        crew_actions = [a for a in result["actions"] if a["action_type"] == "crew_reassign"]
        assert len(crew_actions) == 1
        assert crew_actions[0]["feasible"] is True

    def test_cancel_when_no_crew(self):
        flight = _make_flight()
        result = self._run([flight], [])
        cancels = [a for a in result["actions"] if a["action_type"] == "flight_cancel"]
        assert len(cancels) == 1

    def test_duty_limit_respected(self):
        flight = _make_flight(duration_h=5)
        # captain with 10h already flown, max 14h — can't add 5h
        over_duty_capt = _make_crew("CPT999", "Tired Capt", "captain", duty=10.0, max_duty=14.0)
        fresh_capt     = _make_crew("CPT001", "Fresh Capt", "captain", duty=2.0,  max_duty=14.0)
        fo             = _make_crew("FO001",  "FO",         "first_officer", duty=2.0)
        fas            = [_make_crew(f"FA{i}", f"FA{i}", "flight_attendant", duty=1.0)
                          for i in range(3)]
        crew = [over_duty_capt, fresh_capt, fo] + fas
        result = self._run([flight], crew)
        # should succeed using fresh captain, not tired one
        crew_actions = [a for a in result["actions"] if a["action_type"] == "crew_reassign"]
        if crew_actions and crew_actions[0]["feasible"]:
            assignments = crew_actions[0]["metadata"]["assignments"]
            capt_assign = [a for a in assignments if a["role"] == "captain"]
            assert capt_assign[0]["employee_id"] == "CPT001"

    def test_qualification_required(self):
        flight  = _make_flight()
        type_map = {flight["id"]: "B777"}
        crew = [
            _make_crew("CPT001", "Alice", "captain",       quals=["A320"]),  # not B777
            _make_crew("FO001",  "Bob",   "first_officer", quals=["A320"]),
        ]
        result = self._run([flight], crew, type_map)
        # neither qualified for B777 → cancel
        cancels = [a for a in result["actions"] if a["action_type"] == "flight_cancel"]
        assert len(cancels) >= 1

    def test_no_double_crew_assignment(self):
        f1 = _make_flight(flight_id="f1", fn="AA101")
        f2 = _make_flight(flight_id="f2", fn="AA102")
        crew = self._full_crew()   # just enough for one flight
        result = self._run([f1, f2], crew)
        # second flight should be partial or cancelled
        cancels = [a for a in result["actions"] if a["action_type"] == "flight_cancel"]
        partials = [a for a in result["actions"]
                    if a["action_type"] == "crew_reassign" and not a["feasible"]]
        assert len(cancels) + len(partials) >= 1

    def test_rest_status_crew_excluded(self):
        flight = _make_flight()
        resting_capt = _make_crew("CPT_REST", "Resting", "captain", status="rest")
        result = self._run([flight], [resting_capt])
        cancels = [a for a in result["actions"] if a["action_type"] == "flight_cancel"]
        assert len(cancels) == 1


# ── PassengerAgent tests ──────────────────────────────────────────────────────

class TestPassengerAgent:

    def _run(self, disrupted, all_flights, passengers, itineraries):
        from agents.passenger_agent import PassengerAgent
        ctx = {
            "disruption":       _make_disruption(flight_id=disrupted[0]["id"]),
            "affected_flights": disrupted,
            "all_flights":      all_flights,
            "passengers":       passengers,
            "itineraries":      itineraries,
        }
        return PassengerAgent().run(ctx)

    def test_rebook_on_alternative(self):
        disrupted = _make_flight(flight_id="f1", origin="ATL", dest="ORD")
        alt       = _make_flight(flight_id="f2", fn="AA102", origin="ATL", dest="ORD",
                                  status="scheduled", delay=0, booked=50, dep_offset_h=3)
        pax  = _make_passenger()
        itin = _make_itin(flight_id="f1", pax_id="p1")
        result = self._run([disrupted], [disrupted, alt], [pax], [itin])
        rebooks = [a for a in result["actions"] if a["feasible"]]
        assert len(rebooks) >= 1

    def test_stranded_when_no_alternative(self):
        disrupted = _make_flight(flight_id="f1", origin="ATL", dest="ORD")
        pax  = _make_passenger()
        itin = _make_itin(flight_id="f1", pax_id="p1")
        result = self._run([disrupted], [disrupted], [pax], [itin])
        stranded = [a for a in result["actions"] if not a["feasible"]]
        assert len(stranded) >= 1

    def test_first_class_before_economy(self):
        from agents.passenger_agent import _priority_score
        first_pax = _make_passenger(pax_id="p1", fare="first")
        econ_pax  = _make_passenger(pax_id="p2", fare="economy")
        itin = _make_itin()
        assert _priority_score(first_pax, itin) > _priority_score(econ_pax, itin)

    def test_frequent_flyer_priority(self):
        from agents.passenger_agent import _priority_score
        ff     = _make_passenger(pax_id="p1", ff=True)
        non_ff = _make_passenger(pax_id="p2", ff=False)
        itin = _make_itin()
        assert _priority_score(ff, itin) > _priority_score(non_ff, itin)

    def test_high_misconnect_risk_priority(self):
        from agents.passenger_agent import _priority_score
        pax = _make_passenger()
        itin_high = _make_itin(risk=0.9, conn_time=20)
        itin_low  = _make_itin(risk=0.1, conn_time=120)
        assert _priority_score(pax, itin_high) > _priority_score(pax, itin_low)

    def test_full_flight_not_overbooked(self):
        disrupted = _make_flight(flight_id="f1", origin="ATL", dest="ORD")
        alt = _make_flight(flight_id="f2", fn="AA102", origin="ATL", dest="ORD",
                            status="scheduled", capacity=2, booked=2, dep_offset_h=3)
        pax1  = _make_passenger(pax_id="p1", pnr="AAA111")
        pax2  = _make_passenger(pax_id="p2", pnr="BBB222")
        itin1 = _make_itin(itin_id="i1", pax_id="p1", flight_id="f1")
        itin2 = _make_itin(itin_id="i2", pax_id="p2", flight_id="f1")
        result = self._run([disrupted], [disrupted, alt], [pax1, pax2], [itin1, itin2])
        stranded = [a for a in result["actions"] if not a["feasible"]]
        assert len(stranded) == 2   # alt is full — both stranded

    def test_no_rebook_to_disrupted_flight(self):
        disrupted = _make_flight(flight_id="f1", origin="ATL", dest="ORD")
        pax  = _make_passenger()
        itin = _make_itin(flight_id="f1")
        # only disrupted flight available — should not rebook onto itself
        result = self._run([disrupted], [disrupted], [pax], [itin])
        for a in result["actions"]:
            if a["feasible"]:
                assert a["metadata"].get("new_flight_id") != "f1"


# ── CoordinatorAgent tests ────────────────────────────────────────────────────

class TestCoordinatorAgent:

    def _make_fleet_proposal(self, flight_id="f1", tail="N-TST1", atype="B737",
                               feasible=True, conflict=False):
        return {
            "agent": "fleet",
            "cost": 0.1,
            "confidence": 0.9,
            "notes": "",
            "actions": [{
                "action_type": "aircraft_swap", "agent_source": "fleet",
                "flight_id": flight_id, "flight_number": "AA101",
                "aircraft_id": "ac1",
                "description": f"Swap to {tail}",
                "cost_score": 0.1, "impact_score": 0.8,
                "feasible": feasible, "conflict_flag": conflict,
                "metadata": {"new_tail": tail, "new_type": atype,
                              "needs_reposition": False, "reposition_from": None},
            }],
        }

    def _make_crew_proposal(self, flight_id="f1", emp_id="CPT001", feasible=True):
        return {
            "agent": "crew", "cost": 0.05, "confidence": 0.8, "notes": "",
            "actions": [{
                "action_type": "crew_reassign", "agent_source": "crew",
                "flight_id": flight_id, "flight_number": "AA101",
                "description": "Assign crew",
                "cost_score": 0.05, "impact_score": 0.9,
                "feasible": feasible, "conflict_flag": False,
                "metadata": {
                    "assignments": [{"employee_id": emp_id, "name": "Alice",
                                      "role": "captain", "cost": 0.05,
                                      "needs_reposition": False}],
                    "fully_crewed": feasible,
                    "partial_notes": [],
                },
            }],
        }

    def _make_pax_proposal(self, flight_id="f1", pax_id="p1", feasible=True,
                             misconnect_risk=0.5, delay_min=60):
        return {
            "agent": "passenger", "cost": 0.05, "confidence": 0.75, "notes": "",
            "actions": [{
                "action_type": "passenger_rebook", "agent_source": "passenger",
                "flight_id": flight_id, "flight_number": "AA101",
                "passenger_id": pax_id,
                "description": "Rebook pax",
                "cost_score": 0.05, "impact_score": 0.7,
                "feasible": feasible, "conflict_flag": False,
                "metadata": {"misconnect_risk": misconnect_risk,
                              "delay_minutes": delay_min, "pnr": "ABC123",
                              "fare_class": "economy", "frequent_flyer": False},
            }],
        }

    def test_merged_plan_has_all_action_types(self):
        from agents.coordinator import CoordinatorAgent
        proposals = [
            self._make_fleet_proposal(),
            self._make_crew_proposal(),
            self._make_pax_proposal(),
        ]
        plan = CoordinatorAgent().run(proposals, _make_disruption())
        types = {a["action_type"] for a in plan["final_actions_json"]}
        assert "aircraft_swap"   in types
        assert "crew_reassign"   in types
        assert "passenger_rebook" in types

    def test_conflict_detected_double_tail(self):
        from agents.coordinator import CoordinatorAgent
        p1 = self._make_fleet_proposal(flight_id="f1", tail="N-SAME")
        p2 = self._make_fleet_proposal(flight_id="f2", tail="N-SAME")
        p2["agent"] = "fleet"
        proposals = [
            {"agent": "fleet", "cost": 0.1, "confidence": 0.9, "notes": "",
             "actions": p1["actions"] + p2["actions"]},
            self._make_crew_proposal(),
            self._make_pax_proposal(),
        ]
        plan = CoordinatorAgent().run(proposals, _make_disruption())
        conflicts = [a for a in plan["final_actions_json"] if a["conflict_flag"]]
        assert len(conflicts) >= 1

    def test_cancellation_wins_over_swap(self):
        from agents.coordinator import CoordinatorAgent
        cancel_action = {
            "action_type": "flight_cancel", "agent_source": "crew",
            "flight_id": "f1", "flight_number": "AA101",
            "description": "Cancel f1", "cost_score": 1.0, "impact_score": 0.8,
            "feasible": True, "conflict_flag": False, "metadata": {},
        }
        crew_p = self._make_crew_proposal(flight_id="f1")
        crew_p["actions"].append(cancel_action)

        proposals = [
            self._make_fleet_proposal(flight_id="f1"),
            crew_p,
            self._make_pax_proposal(flight_id="f1"),
        ]
        plan = CoordinatorAgent().run(proposals, _make_disruption())

        # aircraft_swap for f1 should be marked infeasible (cancel wins)
        swap_f1 = [a for a in plan["final_actions_json"]
                    if a["action_type"] == "aircraft_swap" and a["flight_id"] == "f1"]
        if swap_f1:
            assert not swap_f1[0]["feasible"]

    def test_metrics_computed(self):
        from agents.coordinator import CoordinatorAgent
        proposals = [
            self._make_fleet_proposal(),
            self._make_crew_proposal(),
            self._make_pax_proposal(misconnect_risk=0.8, delay_min=90),
        ]
        plan = CoordinatorAgent().run(proposals, _make_disruption())
        assert plan["cancellations_avoided"]  >= 0
        assert plan["misconnects_avoided"]    >= 0
        assert plan["total_delay_reduction"]  >= 0

    def test_summary_keys_present(self):
        from agents.coordinator import CoordinatorAgent
        proposals = [
            self._make_fleet_proposal(),
            self._make_crew_proposal(),
            self._make_pax_proposal(),
        ]
        plan = CoordinatorAgent().run(proposals, _make_disruption())
        required = {"total_actions","by_action_type","cancellations_avoided",
                    "misconnects_avoided","conflict_count","feasible_actions"}
        assert required.issubset(plan["summary"].keys())

    def test_empty_proposals_handled(self):
        from agents.coordinator import CoordinatorAgent
        proposals = [
            {"agent": "fleet",     "actions": [], "cost": 0, "confidence": 1, "notes": ""},
            {"agent": "crew",      "actions": [], "cost": 0, "confidence": 1, "notes": ""},
            {"agent": "passenger", "actions": [], "cost": 0, "confidence": 1, "notes": ""},
        ]
        plan = CoordinatorAgent().run(proposals, _make_disruption())
        assert plan["final_actions_json"] == []
        assert plan["cancellations_avoided"] == 0


# ── Full pipeline integration test ────────────────────────────────────────────

class TestFullPipeline:

    def test_pipeline_end_to_end(self):
        from agents.coordinator import run_recovery_pipeline
        dep = datetime.utcnow() + timedelta(hours=3)
        arr = dep + timedelta(hours=2)
        dep2 = dep + timedelta(hours=1)
        arr2 = dep2 + timedelta(hours=2)

        disrupted = _make_flight(flight_id="f1", origin="ATL", dest="ORD")
        alt       = {**_make_flight(flight_id="f2", fn="AA200", origin="ATL",
                                     dest="ORD", status="scheduled", delay=0,
                                     booked=50, dep_offset_h=4),}

        crew = [
            _make_crew("CPT001", "Alice", "captain",          quals=["B737"]),
            _make_crew("FO001",  "Bob",   "first_officer",    quals=["B737"]),
            _make_crew("FA001",  "Carol", "flight_attendant", quals=["B737"]),
            _make_crew("FA002",  "Dave",  "flight_attendant", quals=["B737"]),
        ]
        fleet = [_make_aircraft(current="ATL", status="available")]
        pax   = [_make_passenger()]
        itins = [_make_itin(flight_id="f1")]
        disruption = _make_disruption(flight_id="f1")

        plan = run_recovery_pipeline(
            disruption=disruption,
            affected_flights=[disrupted],
            all_flights=[disrupted, alt],
            available_crew=crew,
            available_aircraft=fleet,
            passengers=pax,
            itineraries=itins,
        )

        assert "final_actions_json" in plan
        assert "summary" in plan
        assert isinstance(plan["final_actions_json"], list)
        assert plan["summary"]["total_actions"] >= 1