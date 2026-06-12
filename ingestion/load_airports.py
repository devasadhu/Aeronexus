"""
ingestion/load_airports.py

Downloads OurAirports datasets, seeds the airports and routes tables,
and builds + serialises a NetworkX flight-network graph.

Usage:
    python -m ingestion.load_airports [--hub-only] [--save-graph graph.gpickle]
"""

import os
import sys
import csv
import math
import logging
import argparse
import pickle
import io
from pathlib import Path
from datetime import datetime

import requests
import networkx as nx

# ── allow running from repo root ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── OurAirports URLs ──────────────────────────────────────────────────────────
AIRPORTS_URL = "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/airports.csv"
RUNWAYS_URL  = "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/runways.csv"
ROUTES_URL   = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/routes.dat"

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Major hubs (IATA codes) used to filter routes to a manageable subset ─────
MAJOR_HUBS = {
    # North America
    "ATL","ORD","LAX","DFW","DEN","JFK","SFO","LAS","SEA","MCO",
    "MIA","CLT","PHX","EWR","IAH","BOS","MSP","DTW","FLL","PHL",
    # Europe
    "LHR","CDG","AMS","FRA","MAD","BCN","FCO","MUC","ZRH","LIS",
    # Asia-Pacific
    "DXB","SIN","HKG","NRT","ICN","PEK","PVG","BKK","SYD","KUL",
    # India
    "DEL","BOM","BLR","MAA","HYD","CCU",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_csv(url: str, cache_name: str) -> str:
    cache_path = DATA_DIR / cache_name
    if cache_path.exists():
        log.info(f"Using cached {cache_name}")
        return cache_path.read_text(encoding="utf-8", errors="replace")
    log.info(f"Downloading {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    cache_path.write_bytes(r.content)
    log.info(f"Saved to {cache_path}")
    return r.text


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_airports(hub_only: bool = False) -> dict:
    """
    Parse OurAirports airports.csv.
    Returns dict: IATA -> airport_dict
    Only keeps large/medium airports with valid IATA codes.
    """
    raw = fetch_csv(AIRPORTS_URL, "airports.csv")
    reader = csv.DictReader(io.StringIO(raw))
    airports = {}

    for row in reader:
        iata = (row.get("iata_code") or "").strip().upper()
        atype = (row.get("type") or "").strip()

        if not iata or len(iata) != 3:
            continue
        if atype not in ("large_airport", "medium_airport"):
            continue
        if hub_only and iata not in MAJOR_HUBS:
            continue

        try:
            lat = float(row["latitude_deg"])
            lon = float(row["longitude_deg"])
        except (ValueError, KeyError):
            continue

        airports[iata] = {
            "id":          iata,
            "icao":        (row.get("gps_code") or "").strip() or None,
            "name":        row.get("name", ""),
            "city":        row.get("municipality", ""),
            "country":     row.get("iso_country", ""),
            "latitude":    lat,
            "longitude":   lon,
            "altitude_ft": int(float(row.get("elevation_ft") or 0)),
            "timezone":    None,
            "is_hub":      iata in MAJOR_HUBS,
        }

    log.info(f"Parsed {len(airports)} airports")
    return airports


def load_routes(airports: dict) -> list:
    """
    Parse OpenFlights routes.dat (comma-separated, no header).
    Columns: airline,airline_id,src,src_id,dst,dst_id,codeshare,stops,equipment
    Only keeps direct (0-stop) routes between airports in our set.
    Returns list of route dicts.
    """
    raw = fetch_csv(ROUTES_URL, "routes.dat")
    routes = []
    seen = set()

    for line in raw.splitlines():
        parts = line.split(",")
        if len(parts) < 9:
            continue

        airline = parts[0].strip()
        src     = parts[2].strip().upper()
        dst     = parts[4].strip().upper()
        stops   = parts[7].strip()

        if stops != "0":
            continue
        if src not in airports or dst not in airports:
            continue

        key = (src, dst)
        if key in seen:
            continue
        seen.add(key)

        src_a = airports[src]
        dst_a = airports[dst]
        dist  = haversine_km(src_a["latitude"], src_a["longitude"],
                              dst_a["latitude"], dst_a["longitude"])
        # approximate cruise speed 850 km/h
        duration_min = int((dist / 850) * 60 + 30)   # +30 min taxi/approach

        routes.append({
            "origin_id":         src,
            "destination_id":    dst,
            "airline_code":      airline,
            "distance_km":       round(dist, 1),
            "avg_duration_min":  duration_min,
            "is_active":         True,
        })

    log.info(f"Parsed {len(routes)} routes")
    return routes


# ── NetworkX Graph ────────────────────────────────────────────────────────────

def build_graph(airports: dict, routes: list) -> nx.DiGraph:
    """
    Directed weighted graph.
    Nodes: airport IATA codes, with lat/lon/hub attributes.
    Edges: (src, dst) with distance_km, duration_min, airline_code.
    """
    G = nx.DiGraph()

    for iata, a in airports.items():
        G.add_node(iata,
                   name=a["name"],
                   city=a["city"],
                   country=a["country"],
                   lat=a["latitude"],
                   lon=a["longitude"],
                   is_hub=a["is_hub"])

    for r in routes:
        G.add_edge(
            r["origin_id"], r["destination_id"],
            distance_km=r["distance_km"],
            duration_min=r["avg_duration_min"],
            airline=r["airline_code"],
            weight=r["avg_duration_min"],   # for shortest-path queries
        )

    log.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def graph_stats(G: nx.DiGraph):
    """Print a few useful network stats."""
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    log.info("Top 10 most connected airports:")
    for iata, deg in degrees[:10]:
        log.info(f"  {iata:5s}  degree={deg}")

    hubs = [n for n, d in G.nodes(data=True) if d.get("is_hub")]
    log.info(f"Hub airports in graph: {len(hubs)}")

    # connectivity
    wcc = nx.number_weakly_connected_components(G)
    log.info(f"Weakly connected components: {wcc}")


# ── DB Seed ───────────────────────────────────────────────────────────────────

def seed_db(airports: dict, routes: list):
    """Insert airports and routes into PostgreSQL via SQLAlchemy."""
    try:
        from backend.db import get_db_ctx
        from backend.models import Airport, Route
        import uuid
    except ImportError:
        log.warning("Could not import backend modules. Skipping DB seed.")
        log.warning("Run from repo root: python -m ingestion.load_airports")
        return

    with get_db_ctx() as db:
        # upsert airports
        existing_airports = {a.id for a in db.query(Airport.id).all()}
        new_airports = [Airport(**a) for iata, a in airports.items()
                        if iata not in existing_airports]
        db.bulk_save_objects(new_airports)
        db.flush()
        log.info(f"Inserted {len(new_airports)} new airports")

        # upsert routes (check by origin+dest pair)
        existing_routes = {
            (r.origin_id, r.destination_id)
            for r in db.query(Route.origin_id, Route.destination_id).all()
        }
        new_routes = []
        for r in routes:
            if (r["origin_id"], r["destination_id"]) not in existing_routes:
                new_routes.append(Route(id=str(uuid.uuid4()), **r))

        db.bulk_save_objects(new_routes)
        log.info(f"Inserted {len(new_routes)} new routes")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-only",   action="store_true",
                        help="Only load major hub airports (faster)")
    parser.add_argument("--save-graph", default="data/processed/flight_graph.gpickle",
                        help="Path to save the NetworkX graph")
    parser.add_argument("--no-db",      action="store_true",
                        help="Skip DB seed (graph + CSV only)")
    args = parser.parse_args()

    airports = load_airports(hub_only=args.hub_only)
    routes   = load_routes(airports)
    G        = build_graph(airports, routes)

    graph_stats(G)

    # save graph
    graph_path = Path(args.save_graph)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    log.info(f"Graph saved to {graph_path}")

    if not args.no_db:
        seed_db(airports, routes)
    else:
        log.info("Skipping DB seed (--no-db)")


if __name__ == "__main__":
    main()