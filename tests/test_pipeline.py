"""
tests/test_pipeline.py

Covers:
  - load_airports: graph properties
  - synthetic generators: schema correctness
  - load_bts: normalisation + disruption detection
  - feature_builder: feature extraction correctness
  - cascade_model: predict_cascade output shape + constraints
  - severity_scorer: scoring + severity label assignment
"""

import sys, json, pickle, pytest
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def graph():
    path = Path("data/processed/flight_graph.gpickle")
    assert path.exists(), "Run load_airports.py --hub-only first"
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def flights():
    path = Path("data/processed/flights.json")
    assert path.exists(), "Run load_bts.py --synthetic first"
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def disruptions():
    path = Path("data/processed/disruptions_seed.json")
    assert path.exists(), "Run load_bts.py --synthetic first"
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def crew():
    path = Path("data/processed/crew_roster.json")
    assert path.exists(), "Run gen_crew.py first"
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def fleet():
    path = Path("data/processed/aircraft_fleet.json")
    assert path.exists(), "Run gen_aircraft.py first"
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def passengers():
    path = Path("data/processed/passengers.json")
    assert path.exists(), "Run gen_passengers.py first"
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def itineraries():
    path = Path("data/processed/itineraries.json")
    assert path.exists(), "Run gen_passengers.py first"
    return json.loads(path.read_text())


# ──────────────────────────────────────────────────────────────────────────────
# Graph tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGraph:
    def test_has_nodes(self, graph):
        assert graph.number_of_nodes() >= 40, "Should have at least 40 hub airports"

    def test_has_edges(self, graph):
        assert graph.number_of_edges() >= 500, "Should have at least 500 routes"

    def test_single_component(self, graph):
        import networkx as nx
        assert nx.number_weakly_connected_components(graph) == 1

    def test_known_hubs_present(self, graph):
        for hub in ["ATL", "JFK", "LHR", "DXB", "SIN"]:
            assert hub in graph.nodes, f"{hub} should be in graph"

    def test_edge_has_distance(self, graph):
        for u, v, data in list(graph.edges(data=True))[:20]:
            assert "distance_km" in data
            assert data["distance_km"] > 0

    def test_shortest_path_exists(self, graph):
        import networkx as nx
        path = nx.shortest_path(graph, "ATL", "SIN", weight="weight")
        assert len(path) >= 2

    def test_no_self_loops(self, graph):
        import networkx as nx
        self_loops = list(nx.selfloop_edges(graph))
        assert len(self_loops) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCrewRoster:
    def test_count(self, crew):
        assert len(crew) >= 60

    def test_required_fields(self, crew):
        required = {"id","employee_id","name","role","base_airport_id",
                    "status","max_duty_hours","current_duty_hours","qualifications_json"}
        for c in crew:
            assert required.issubset(c.keys()), f"Missing fields in crew: {c['employee_id']}"

    def test_roles(self, crew):
        roles = {c["role"] for c in crew}
        assert "captain" in roles
        assert "first_officer" in roles
        assert "flight_attendant" in roles

    def test_duty_hours_within_limits(self, crew):
        # Only on_duty crew should be within their duty limit.
        # Crew in "rest" status may show hours above max (just came off a long duty).
        for c in crew:
            if c["status"] == "on_duty":
                assert c["current_duty_hours"] <= c["max_duty_hours"] + 0.1, \
                    f"{c['employee_id']} on-duty hours exceed max"
            assert c["current_duty_hours"] >= 0

    def test_qualifications_non_empty(self, crew):
        for c in crew:
            if c["role"] in ("captain", "first_officer"):
                assert len(c["qualifications_json"]) >= 1

    def test_unique_employee_ids(self, crew):
        ids = [c["employee_id"] for c in crew]
        assert len(ids) == len(set(ids)), "Duplicate employee IDs found"


class TestFleet:
    def test_count(self, fleet):
        assert len(fleet) >= 20

    def test_required_fields(self, fleet):
        required = {"id","tail_number","aircraft_type","capacity","status",
                    "home_airport_id","maintenance_windows_json"}
        for a in fleet:
            assert required.issubset(a.keys())

    def test_valid_statuses(self, fleet):
        valid = {"available","in_flight","maintenance","aog"}
        for a in fleet:
            assert a["status"] in valid

    def test_unique_tail_numbers(self, fleet):
        tails = [a["tail_number"] for a in fleet]
        assert len(tails) == len(set(tails))

    def test_capacity_realistic(self, fleet):
        for a in fleet:
            assert 50 <= a["capacity"] <= 600

    def test_aog_has_maintenance_window(self, fleet):
        aog = [a for a in fleet if a["status"] == "aog"]
        for a in aog:
            windows = a["maintenance_windows_json"]
            assert len(windows) >= 1
            aog_windows = [w for w in windows if w["type"] == "AOG"]
            assert len(aog_windows) >= 1


class TestPassengers:
    def test_count(self, passengers):
        assert len(passengers) >= 100

    def test_unique_pnrs(self, passengers):
        pnrs = [p["pnr"] for p in passengers]
        assert len(pnrs) == len(set(pnrs))

    def test_fare_classes(self, passengers):
        classes = {p["fare_class"] for p in passengers}
        assert "economy" in classes

    def test_itinerary_links(self, passengers, itineraries):
        pax_ids = {p["id"] for p in passengers}
        for itin in itineraries:
            assert itin["passenger_id"] in pax_ids

    def test_misconnect_risk_range(self, itineraries):
        for i in itineraries:
            assert 0.0 <= i["misconnect_risk"] <= 1.0

    def test_leg_numbers_positive(self, itineraries):
        for i in itineraries:
            assert i["leg_number"] >= 1


# ──────────────────────────────────────────────────────────────────────────────
# BTS / flight data tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFlightData:
    def test_count(self, flights):
        assert len(flights) >= 1000

    def test_required_fields(self, flights):
        required = {"id","flight_number","airline_code","origin_id","destination_id",
                    "scheduled_departure","scheduled_arrival","status","delay_minutes"}
        for f in flights[:100]:
            assert required.issubset(f.keys())

    def test_valid_statuses(self, flights):
        valid = {"scheduled","departed","arrived","delayed","cancelled","diverted"}
        for f in flights:
            assert f["status"] in valid

    def test_no_negative_delays(self, flights):
        for f in flights:
            assert f["delay_minutes"] >= 0

    def test_cancelled_have_no_actual_times(self, flights):
        cancelled = [f for f in flights if f["status"] == "cancelled"]
        for f in cancelled[:20]:
            assert f.get("actual_departure") is None

    def test_disruptions_are_subset(self, flights, disruptions):
        flight_ids = {f["id"] for f in flights}
        for d in disruptions:
            assert d["flight_id"] in flight_ids

    def test_disruption_types(self, disruptions):
        valid_types = {"weather","carrier","mechanical","atc","airport","unknown"}
        for d in disruptions:
            assert d["type"] in valid_types

    def test_origin_ne_destination(self, flights):
        for f in flights[:200]:
            assert f["origin_id"] != f["destination_id"]


# ──────────────────────────────────────────────────────────────────────────────
# Feature builder tests
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatureBuilder:
    def test_extract_features_shape(self, flights, disruptions, graph):
        from ml.feature_builder import (
            extract_features, compute_centrality,
            build_historical_rates, build_downstream_index
        )
        centrality     = compute_centrality(graph)
        hist_rates     = build_historical_rates(flights)
        downstream_idx = build_downstream_index(flights)

        flight     = flights[0]
        disruption = disruptions[0]
        feats      = extract_features(flight, disruption, graph,
                                       centrality, hist_rates, downstream_idx)

        assert isinstance(feats, dict)
        assert len(feats) >= 25

    def test_feature_ranges(self, flights, disruptions, graph):
        from ml.feature_builder import (
            extract_features, compute_centrality,
            build_historical_rates, build_downstream_index
        )
        centrality     = compute_centrality(graph)
        hist_rates     = build_historical_rates(flights)
        downstream_idx = build_downstream_index(flights)

        for f, d in zip(flights[:10], disruptions[:10]):
            feats = extract_features(f, d, graph, centrality,
                                      hist_rates, downstream_idx)
            assert 0 <= feats["hour"]        <= 23
            assert 0 <= feats["day_of_week"] <= 6
            assert 1 <= feats["month"]       <= 12
            assert feats["is_peak_hour"]   in (0, 1)
            assert feats["is_weekend"]     in (0, 1)
            assert feats["bc_origin"]      >= 0
            assert feats["delay_minutes"]  >= 0

    def test_hub_flags_correct(self, flights, disruptions, graph):
        from ml.feature_builder import (
            extract_features, compute_centrality,
            build_historical_rates, build_downstream_index, HUB_AIRPORTS
        )
        centrality     = compute_centrality(graph)
        hist_rates     = build_historical_rates(flights)
        downstream_idx = build_downstream_index(flights)

        for f in flights[:50]:
            feats = extract_features(f, disruptions[0], graph,
                                      centrality, hist_rates, downstream_idx)
            expected_hub = int(f["origin_id"] in HUB_AIRPORTS)
            assert feats["is_hub_origin"] == expected_hub


# ──────────────────────────────────────────────────────────────────────────────
# Cascade model tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCascadeModel:
    def test_model_loads(self):
        from ml.cascade_model import load_model
        model = load_model()
        assert model is not None

    def test_predict_returns_list(self, flights, disruptions, graph):
        from ml.cascade_model import predict_cascade
        from ml.feature_builder import build_historical_rates, build_downstream_index

        hist_rates     = build_historical_rates(flights)
        downstream_idx = build_downstream_index(flights)

        disruption = disruptions[0]
        candidates = [f for f in flights
                       if f["origin_id"] == disruption.get("destination", "ATL")][:10]
        results = predict_cascade(disruption, candidates, graph,
                                   hist_rates, downstream_idx, risk_threshold=0.1)
        assert isinstance(results, list)

    def test_risk_scores_in_range(self, flights, disruptions, graph):
        from ml.cascade_model import predict_cascade
        from ml.feature_builder import build_historical_rates, build_downstream_index

        hist_rates     = build_historical_rates(flights)
        downstream_idx = build_downstream_index(flights)

        disruption = disruptions[0]
        candidates = flights[:20]
        results = predict_cascade(disruption, candidates, graph,
                                   hist_rates, downstream_idx, risk_threshold=0.0)
        for r in results:
            assert 0.0 <= r["risk_score"] <= 1.0, f"Invalid risk score: {r['risk_score']}"

    def test_sorted_descending(self, flights, disruptions, graph):
        from ml.cascade_model import predict_cascade
        from ml.feature_builder import build_historical_rates, build_downstream_index

        hist_rates     = build_historical_rates(flights)
        downstream_idx = build_downstream_index(flights)

        disruption = disruptions[0]
        candidates = flights[:30]
        results = predict_cascade(disruption, candidates, graph,
                                   hist_rates, downstream_idx, risk_threshold=0.0)
        scores = [r["risk_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_threshold_filters_results(self, flights, disruptions, graph):
        from ml.cascade_model import predict_cascade
        from ml.feature_builder import build_historical_rates, build_downstream_index

        hist_rates     = build_historical_rates(flights)
        downstream_idx = build_downstream_index(flights)

        disruption = disruptions[0]
        candidates = flights[:30]

        results_low  = predict_cascade(disruption, candidates, graph,
                                        hist_rates, downstream_idx, risk_threshold=0.0)
        results_high = predict_cascade(disruption, candidates, graph,
                                        hist_rates, downstream_idx, risk_threshold=0.9)

        assert len(results_low) >= len(results_high)
        for r in results_high:
            assert r["risk_score"] >= 0.9

    def test_empty_candidates(self, disruptions, graph):
        from ml.cascade_model import predict_cascade
        from ml.feature_builder import build_historical_rates, build_downstream_index

        results = predict_cascade(disruptions[0], [], graph, {}, {})
        assert results == []


# ──────────────────────────────────────────────────────────────────────────────
# Severity scorer tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSeverityScorer:
    def _make_affected(self, n: int, risk: float = 0.7) -> list:
        return [{"flight_id": f"f{i}", "flight_number": f"XX{i}",
                 "risk_score": risk, "delay_estimate_min": 45}
                for i in range(n)]

    def test_score_increases_with_delay(self):
        from ml.severity_scorer import compute_severity_score
        low_delay  = compute_severity_score(
            {"type":"weather","delay_minutes":30},  self._make_affected(2), 100, True, 10)
        high_delay = compute_severity_score(
            {"type":"weather","delay_minutes":180}, self._make_affected(2), 100, True, 10)
        assert high_delay["score"] > low_delay["score"]

    def test_hub_penalty_applied(self):
        from ml.severity_scorer import compute_severity_score
        hub     = compute_severity_score({"type":"carrier","delay_minutes":60},
                                          self._make_affected(2), 100, True,  12)
        non_hub = compute_severity_score({"type":"carrier","delay_minutes":60},
                                          self._make_affected(2), 100, False, 12)
        assert hub["score"] > non_hub["score"]

    def test_weather_multiplier_raises_score(self):
        from ml.severity_scorer import compute_severity_score
        weather = compute_severity_score(
            {"type":"weather","delay_minutes":60}, self._make_affected(3), 200, True, 10)
        carrier = compute_severity_score(
            {"type":"carrier","delay_minutes":60}, self._make_affected(3), 200, True, 10)
        assert weather["score"] > carrier["score"]

    def test_severity_labels(self):
        from ml.severity_scorer import score_to_severity
        assert score_to_severity({"score": 90}) == "critical"
        assert score_to_severity({"score": 60}) == "high"
        assert score_to_severity({"score": 35}) == "medium"
        assert score_to_severity({"score": 10}) == "low"

    def test_build_disruption_event_fields(self, flights, disruptions, graph):
        from ml.severity_scorer import build_disruption_event
        from ml.cascade_model import predict_cascade
        from ml.feature_builder import build_historical_rates, build_downstream_index

        flights_lookup = {f["id"]: f for f in flights}
        hist_rates     = build_historical_rates(flights)
        downstream_idx = build_downstream_index(flights)

        disruption = disruptions[0]
        candidates = flights[:20]
        affected   = predict_cascade(disruption, candidates, graph,
                                      hist_rates, downstream_idx, risk_threshold=0.0)

        event = build_disruption_event(disruption, affected, flights_lookup, graph)

        required = {"root_flight_id","disruption_type","severity","severity_score",
                    "affected_flights_json","total_affected_flights","total_affected_pax",
                    "estimated_delay_min","resolved"}
        assert required.issubset(event.keys())
        assert event["severity"] in ("low","medium","high","critical")
        assert 0 <= event["severity_score"] <= 100
        assert event["total_affected_flights"] == len(affected)
        assert event["total_affected_pax"] >= 0

    def test_event_consistent_with_affected_count(self, flights, disruptions, graph):
        from ml.severity_scorer import build_disruption_event
        from ml.cascade_model import predict_cascade
        from ml.feature_builder import build_historical_rates, build_downstream_index

        flights_lookup = {f["id"]: f for f in flights}
        hist_rates     = build_historical_rates(flights)
        downstream_idx = build_downstream_index(flights)

        for d in disruptions[:5]:
            candidates = flights[:15]
            affected = predict_cascade(d, candidates, graph,
                                        hist_rates, downstream_idx, risk_threshold=0.0)
            event = build_disruption_event(d, affected, flights_lookup, graph)
            assert event["total_affected_flights"] == len(event["affected_flights_json"])