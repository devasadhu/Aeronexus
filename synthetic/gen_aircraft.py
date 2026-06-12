"""
synthetic/gen_aircraft.py

Generates synthetic aircraft fleet with realistic:
- Tail numbers and aircraft types
- Current positions and statuses
- Maintenance windows (scheduled + random AOG events)

Usage:
    python -m synthetic.gen_aircraft [--airports ...] [--fleet-size 40] [--no-db]
"""

import sys, uuid, random, json, argparse, logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

random.seed(7)

# ── Aircraft type definitions ─────────────────────────────────────────────────
AIRCRAFT_TYPES = {
    # type:  (capacity, range_km, prefix_pool)
    "B737":  (160, 5765,  ["N", "VT", "AP"]),
    "A320":  (150, 6100,  ["G", "F",  "D" ]),
    "B738":  (162, 5765,  ["N", "VT", "9V"]),
    "A321":  (180, 7400,  ["G", "EC", "I" ]),
    "B777":  (396, 13500, ["A6","VT", "9V"]),
    "B787":  (296, 14140, ["N", "VT", "JA"]),
    "A330":  (260, 13430, ["F", "D",  "A6"]),
    "A350":  (369, 15000, ["G", "9V", "JA"]),
}

FLEET_COMPOSITION = [
    ("B737", 0.25),
    ("A320", 0.20),
    ("B738", 0.10),
    ("A321", 0.10),
    ("B777", 0.10),
    ("B787", 0.10),
    ("A330", 0.10),
    ("A350", 0.05),
]


def random_tail(aircraft_type: str, index: int) -> str:
    _, _, prefixes = AIRCRAFT_TYPES[aircraft_type]
    prefix = random.choice(prefixes)
    suffix = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=3))
    return f"{prefix}-{suffix}{index % 10}"


def random_maintenance_windows(status: str, n: int = 2) -> list:
    """
    Generate 0–3 maintenance windows.
    AOG aircraft have one immediate window starting now.
    """
    windows = []
    now = datetime.utcnow()

    if status == "aog":
        # unscheduled grounding: 4–48 hours from now
        end = now + timedelta(hours=random.randint(4, 48))
        windows.append({
            "start": now.isoformat(),
            "end":   end.isoformat(),
            "type":  "AOG",
            "reason": random.choice(["hydraulic leak","avionics fault","bird strike","engine inspection"]),
        })

    # future scheduled maintenance windows
    for _ in range(random.randint(0, n)):
        offset_days = random.randint(1, 30)
        start = now + timedelta(days=offset_days, hours=random.randint(0, 12))
        duration_hrs = random.choice([4, 8, 12, 24, 48])
        end   = start + timedelta(hours=duration_hrs)
        windows.append({
            "start": start.isoformat(),
            "end":   end.isoformat(),
            "type":  "scheduled",
            "reason": random.choice(["A-check","B-check","avionics update","landing gear inspection"]),
        })

    return windows


def make_aircraft(aircraft_type: str, index: int, home_airport: str, airports: list) -> dict:
    cap, range_km, _ = AIRCRAFT_TYPES[aircraft_type]

    # status distribution
    status = random.choices(
        ["available", "in_flight", "maintenance", "aog"],
        weights=[0.55, 0.25, 0.15, 0.05]
    )[0]

    # current location: in_flight aircraft are "between" airports
    if status == "in_flight":
        current_airport = None   # actually airborne
    elif status in ("maintenance", "aog"):
        current_airport = home_airport
    else:
        # 70% at home, 30% at another hub
        current_airport = home_airport if random.random() < 0.70 else random.choice(airports)

    return {
        "id":                       str(uuid.uuid4()),
        "tail_number":              random_tail(aircraft_type, index),
        "aircraft_type":            aircraft_type,
        "capacity":                 cap + random.randint(-10, 10),   # slight config variance
        "range_km":                 range_km,
        "status":                   status,
        "home_airport_id":          home_airport,
        "current_airport_id":       current_airport,
        "maintenance_windows_json": random_maintenance_windows(status),
    }


def generate_fleet(airports: list, fleet_size: int) -> list:
    fleet = []
    type_counts = {}

    for i in range(fleet_size):
        # pick aircraft type by composition weights
        types, weights = zip(*FLEET_COMPOSITION)
        atype = random.choices(types, weights=weights)[0]
        home  = airports[i % len(airports)]

        aircraft = make_aircraft(atype, i + 1, home, airports)
        fleet.append(aircraft)
        type_counts[atype] = type_counts.get(atype, 0) + 1

    log.info(f"Generated {len(fleet)} aircraft")
    for atype, cnt in sorted(type_counts.items()):
        log.info(f"  {atype}: {cnt}")

    statuses = {}
    for a in fleet:
        statuses[a["status"]] = statuses.get(a["status"], 0) + 1
    log.info("Status breakdown:")
    for s, cnt in sorted(statuses.items()):
        log.info(f"  {s}: {cnt}")

    return fleet


def save_to_json(fleet: list, path: str = "data/processed/aircraft_fleet.json"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(fleet, indent=2))
    log.info(f"Saved fleet to {p}")


def seed_db(fleet: list):
    try:
        from backend.db import get_db_ctx
        from backend.models import Aircraft
    except ImportError:
        log.warning("Could not import backend. Skipping DB seed.")
        return

    with get_db_ctx() as db:
        existing = {a.tail_number for a in db.query(Aircraft.tail_number).all()}
        new_aircraft = [
            Aircraft(
                id=a["id"],
                tail_number=a["tail_number"],
                aircraft_type=a["aircraft_type"],
                capacity=a["capacity"],
                range_km=a["range_km"],
                status=a["status"],
                home_airport_id=a["home_airport_id"],
                current_airport_id=a["current_airport_id"],
                maintenance_windows_json=a["maintenance_windows_json"],
            )
            for a in fleet if a["tail_number"] not in existing
        ]
        db.bulk_save_objects(new_aircraft)
        log.info(f"Inserted {len(new_aircraft)} aircraft into DB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airports", nargs="+",
                        default=["ATL","ORD","LAX","JFK","LHR","DXB","DEL","SIN"])
    parser.add_argument("--fleet-size", type=int, default=40)
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args()

    fleet = generate_fleet(args.airports, args.fleet_size)
    save_to_json(fleet)

    if not args.no_db:
        seed_db(fleet)
    else:
        log.info("Skipping DB seed (--no-db)")


if __name__ == "__main__":
    main()