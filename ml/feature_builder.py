"""
ml/feature_builder.py

Builds ML features for each disrupted flight to train / run the cascade model.

Feature groups:
  1. Temporal     — hour, day_of_week, month, is_peak_hour, is_weekend
  2. Route        — distance, duration estimate, is_hub_origin, is_hub_dest
  3. Network      — betweenness centrality, degree, shortest path alternatives
  4. Disruption   — delay magnitude, cancellation flag, disruption type encoded
  5. Historical   — historical delay rate on this route (from training data)
  6. Downstream   — number of downstream flights sharing same tail / crew slot
"""

import sys, json, pickle, logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

GRAPH_PATH    = Path("data/processed/flight_graph.gpickle")
FLIGHTS_PATH  = Path("data/processed/flights.json")
DISRUPTIONS_PATH = Path("data/processed/disruptions_seed.json")

HUB_AIRPORTS = {
    "ATL","ORD","LAX","DFW","DEN","JFK","SFO","LHR","CDG","AMS",
    "FRA","DXB","SIN","DEL","BOM","NRT","ICN","EWR","IAH","BOS",
}

PEAK_HOURS = {6, 7, 8, 9, 16, 17, 18, 19, 20}

DISRUPTION_TYPE_MAP = {
    "weather": 0, "carrier": 1, "mechanical": 2,
    "atc": 3, "airport": 4, "unknown": 5,
}


# ── Graph utilities ───────────────────────────────────────────────────────────

def load_graph() -> Optional[nx.DiGraph]:
    if not GRAPH_PATH.exists():
        log.warning(f"Graph not found at {GRAPH_PATH}. Run load_airports.py first.")
        return None
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    log.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_centrality(G: nx.DiGraph) -> dict:
    """Betweenness centrality per airport (cached — expensive to compute)."""
    cache_path = Path("data/processed/centrality.json")
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    log.info("Computing betweenness centrality (may take a moment)...")
    bc = nx.betweenness_centrality(G, normalized=True, weight="weight")
    cache_path.write_text(json.dumps(bc))
    log.info("Centrality cached.")
    return bc


def count_alternate_routes(G: nx.DiGraph, origin: str, dest: str,
                            max_hops: int = 2) -> int:
    """Count simple paths from origin to dest with at most max_hops edges."""
    if origin not in G or dest not in G:
        return 0
    try:
        paths = list(nx.all_simple_paths(G, origin, dest, cutoff=max_hops))
        return len(paths)
    except nx.NetworkXError:
        return 0


# ── Historical delay rates ─────────────────────────────────────────────────────

def build_historical_rates(flights: list) -> dict:
    """
    Compute per-route historical delay rate and mean delay from flight data.
    Returns dict: (origin, dest) -> {"delay_rate": float, "mean_delay": float}
    """
    route_stats = {}
    for f in flights:
        key = (f["origin_id"], f["destination_id"])
        if key not in route_stats:
            route_stats[key] = {"total": 0, "delayed": 0, "total_delay": 0.0}
        route_stats[key]["total"] += 1
        if f["delay_minutes"] >= 15:
            route_stats[key]["delayed"] += 1
        route_stats[key]["total_delay"] += f["delay_minutes"]

    result = {}
    for key, stats in route_stats.items():
        n = stats["total"]
        result[key] = {
            "delay_rate":  stats["delayed"] / n if n > 0 else 0.0,
            "mean_delay":  stats["total_delay"] / n if n > 0 else 0.0,
            "sample_size": n,
        }
    return result


# ── Downstream flight count ───────────────────────────────────────────────────

def build_downstream_index(flights: list) -> dict:
    """
    For each airport, count how many departures leave within 3 hours of a given
    scheduled arrival time (potential connections). Keyed by (dest, hour_bucket).
    """
    index = {}
    for f in flights:
        dep = f["scheduled_departure"]
        if isinstance(dep, str):
            try:
                dep = datetime.fromisoformat(dep)
            except ValueError:
                continue
        key = (f["origin_id"], dep.hour)
        index[key] = index.get(key, 0) + 1
    return index


# ── Feature extraction per flight ────────────────────────────────────────────

def extract_features(
    flight: dict,
    disruption: dict,
    G: Optional[nx.DiGraph],
    centrality: dict,
    hist_rates: dict,
    downstream_idx: dict,
) -> dict:
    """
    Build a flat feature dict for one (flight, disruption) pair.
    """
    origin = flight["origin_id"]
    dest   = flight["destination_id"]

    # ── 1. Temporal ──
    dep = flight["scheduled_departure"]
    if isinstance(dep, str):
        dep = datetime.fromisoformat(dep)

    hour        = dep.hour
    dow         = dep.weekday()   # 0=Mon … 6=Sun
    month       = dep.month
    is_peak     = int(hour in PEAK_HOURS)
    is_weekend  = int(dow >= 5)
    is_night    = int(hour < 6 or hour >= 22)

    # ── 2. Route ──
    is_hub_orig = int(origin in HUB_AIRPORTS)
    is_hub_dest = int(dest   in HUB_AIRPORTS)
    both_hubs   = int(is_hub_orig and is_hub_dest)

    edge_data = {}
    if G and G.has_edge(origin, dest):
        edge_data = G[origin][dest]

    distance_km   = edge_data.get("distance_km",   1500.0)
    duration_min  = edge_data.get("duration_min",   180)
    is_long_haul  = int(distance_km > 4000)

    # ── 3. Network ──
    bc_origin = centrality.get(origin, 0.0)
    bc_dest   = centrality.get(dest,   0.0)
    degree_origin = G.degree(origin) if G and origin in G else 0
    degree_dest   = G.degree(dest)   if G and dest   in G else 0
    alt_routes    = count_alternate_routes(G, origin, dest) if G else 0

    # ── 4. Disruption ──
    delay_min     = disruption.get("delay_minutes", 0)
    is_cancelled  = int(disruption.get("type") in ("carrier", "weather") and delay_min == 0)
    is_weather    = int(disruption.get("type") == "weather")
    disruption_type_enc = DISRUPTION_TYPE_MAP.get(disruption.get("type", "unknown"), 5)
    delay_cat     = 0 if delay_min < 30 else (1 if delay_min < 60 else (2 if delay_min < 120 else 3))

    # ── 5. Historical ──
    route_key = (origin, dest)
    hist = hist_rates.get(route_key, {"delay_rate": 0.1, "mean_delay": 15.0, "sample_size": 0})
    hist_delay_rate  = hist["delay_rate"]
    hist_mean_delay  = hist["mean_delay"]
    hist_sample_size = min(hist["sample_size"], 500)   # cap to avoid scale issues

    # ── 6. Downstream pressure ──
    arr = flight["scheduled_arrival"]
    if isinstance(arr, str):
        try:
            arr = datetime.fromisoformat(arr)
            arr_hour = arr.hour
        except ValueError:
            arr_hour = (dep.hour + 2) % 24
    else:
        arr_hour = arr.hour if hasattr(arr, "hour") else 12

    downstream_count = downstream_idx.get((dest, arr_hour), 0)
    downstream_count_next = downstream_idx.get((dest, (arr_hour + 1) % 24), 0)

    return {
        # temporal
        "hour":               hour,
        "day_of_week":        dow,
        "month":              month,
        "is_peak_hour":       is_peak,
        "is_weekend":         is_weekend,
        "is_night":           is_night,
        # route
        "is_hub_origin":      is_hub_orig,
        "is_hub_dest":        is_hub_dest,
        "both_hubs":          both_hubs,
        "distance_km":        distance_km,
        "duration_min":       duration_min,
        "is_long_haul":       is_long_haul,
        # network
        "bc_origin":          round(bc_origin, 5),
        "bc_dest":            round(bc_dest, 5),
        "degree_origin":      degree_origin,
        "degree_dest":        degree_dest,
        "alt_routes":         alt_routes,
        # disruption
        "delay_minutes":      delay_min,
        "delay_category":     delay_cat,
        "is_cancelled":       is_cancelled,
        "is_weather":         is_weather,
        "disruption_type":    disruption_type_enc,
        # historical
        "hist_delay_rate":    round(hist_delay_rate, 4),
        "hist_mean_delay":    round(hist_mean_delay, 2),
        "hist_sample_size":   hist_sample_size,
        # downstream
        "downstream_count":   downstream_count,
        "downstream_next_hr": downstream_count_next,
    }


# ── Build full training dataset ───────────────────────────────────────────────

def build_training_dataset(
    flights: list,
    disruptions: list,
    G: Optional[nx.DiGraph],
    same_aircraft_window_min: int = 180,
) -> pd.DataFrame:
    """
    For each disruption, identify likely impacted downstream flights
    (same origin airport, departing within window_min after disruption).
    Label them as impacted=1. Sample non-impacted flights as negative examples.

    Returns a DataFrame ready for XGBoost training.

    KNOWN LIMITATION (synthetic-data label leakage):
    Positive examples are drawn from flights linked to a disruption
    (delay_minutes > 0 by construction), while negative examples are
    built with a placeholder disruption dict {"delay_minutes": 0,
    "type": "unknown"}. This makes `delay_minutes` and `delay_category`
    perfectly separate the classes (AUC=1.0 on this synthetic set),
    so the trained model is currently learning "was this flight linked
    to a disruption record" rather than genuine cascade-propagation
    signal from network/temporal/historical features.

    To get a model that learns real cascade dynamics, replace this with
    labels derived from actual downstream delay correlation in real BTS
    data (e.g. did flight B's delay increase given flight A's disruption,
    independent of whether B itself was flagged as a disruption).
    """
    centrality    = compute_centrality(G) if G else {}
    hist_rates    = build_historical_rates(flights)
    downstream_idx = build_downstream_index(flights)

    flight_by_id = {f["id"]: f for f in flights}

    # Index flights by (origin, hour) for fast downstream lookup
    dep_index = {}
    for f in flights:
        dep = f["scheduled_departure"]
        if isinstance(dep, str):
            try:
                dep = datetime.fromisoformat(dep)
            except ValueError:
                continue
        key = (f["origin_id"], dep.hour)
        dep_index.setdefault(key, []).append(f)

    rows = []
    disruption_index = {}
    for d in disruptions:
        disruption_index[d["flight_id"]] = d

    for disruption in disruptions:
        root_flight = flight_by_id.get(disruption["flight_id"])
        if not root_flight:
            continue

        dep = root_flight["scheduled_departure"]
        if isinstance(dep, str):
            try:
                dep = datetime.fromisoformat(dep)
            except ValueError:
                continue

        # downstream candidates: same destination airport, next 3 hours
        impacted_origin = root_flight["destination_id"]
        impacted_candidates = []
        for h_offset in range(4):
            hour_key = (impacted_origin, (dep.hour + h_offset) % 24)
            impacted_candidates.extend(dep_index.get(hour_key, []))

        # label them impacted if they share origin with disruption's destination
        impacted_ids = {f["id"] for f in impacted_candidates[:10]}

        # build features for impacted (positive) examples
        for f in impacted_candidates[:5]:
            feats = extract_features(f, disruption, G, centrality,
                                      hist_rates, downstream_idx)
            feats["impacted"] = 1
            feats["risk_score_gt"] = min(1.0,
                0.3 + (disruption.get("delay_minutes", 0) / 300) +
                (0.2 if disruption.get("type") == "weather" else 0))
            rows.append(feats)

        # build features for non-impacted (negative) examples
        non_impacted = [f for f in flights
                         if f["id"] not in impacted_ids
                         and f["origin_id"] != impacted_origin]
        for f in non_impacted[:5]:
            feats = extract_features(f, {"delay_minutes": 0, "type": "unknown"},
                                      G, centrality, hist_rates, downstream_idx)
            feats["impacted"] = 0
            feats["risk_score_gt"] = 0.0
            rows.append(feats)

    df = pd.DataFrame(rows)
    log.info(f"Built training dataset: {len(df)} samples, "
             f"{df['impacted'].sum()} positive, "
             f"{(df['impacted']==0).sum()} negative")
    return df


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    flights     = json.loads(FLIGHTS_PATH.read_text())
    disruptions = json.loads(DISRUPTIONS_PATH.read_text())
    G           = load_graph()

    df = build_training_dataset(flights, disruptions, G)

    out = Path("data/processed/training_features.parquet")
    df.to_parquet(out, index=False)
    log.info(f"Saved training features to {out}")
    log.info(f"Feature columns: {list(df.columns)}")
    print(df.head(3).to_string())