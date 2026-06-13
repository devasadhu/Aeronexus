from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import json
from pathlib import Path

router = APIRouter()

# In-memory store when DB not available (falls back to processed JSON)
_flights_cache = None

def _load_flights():
    global _flights_cache
    if _flights_cache is None:
        p = Path("data/processed/flights.json")
        _flights_cache = json.loads(p.read_text()) if p.exists() else []
    return _flights_cache


@router.get("/", response_model=List[dict])
def list_flights(
    status:  Optional[str] = Query(None),
    origin:  Optional[str] = Query(None),
    dest:    Optional[str] = Query(None),
    limit:   int           = Query(50, le=500),
    offset:  int           = Query(0),
):
    flights = _load_flights()
    if status:
        flights = [f for f in flights if f["status"] == status]
    if origin:
        flights = [f for f in flights if f["origin_id"] == origin.upper()]
    if dest:
        flights = [f for f in flights if f["destination_id"] == dest.upper()]
    return flights[offset: offset + limit]


@router.get("/{flight_id}")
def get_flight(flight_id: str):
    flights = _load_flights()
    for f in flights:
        if f["id"] == flight_id:
            return f
    raise HTTPException(status_code=404, detail="Flight not found")


@router.patch("/{flight_id}")
def update_flight(flight_id: str, update: dict):
    flights = _load_flights()
    for f in flights:
        if f["id"] == flight_id:
            allowed = {"status", "delay_minutes", "actual_departure", "actual_arrival"}
            for k, v in update.items():
                if k in allowed:
                    f[k] = v
            return f
    raise HTTPException(status_code=404, detail="Flight not found")


@router.get("/stats/summary")
def flight_stats():
    flights = _load_flights()
    by_status = {}
    for f in flights:
        s = f["status"]
        by_status[s] = by_status.get(s, 0) + 1
    total_delay = sum(f.get("delay_minutes", 0) for f in flights)
    return {
        "total":          len(flights),
        "by_status":      by_status,
        "avg_delay_min":  round(total_delay / max(len(flights), 1), 1),
    }