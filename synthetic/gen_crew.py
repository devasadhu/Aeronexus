"""
synthetic/gen_crew.py

Generates realistic synthetic crew rosters for a small hub network.
Produces CrewMember + CrewAssignment records.

Usage:
    python -m synthetic.gen_crew [--airports ATL ORD LAX ...] [--count 60] [--no-db]
"""

import sys, uuid, random, json, argparse, logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

random.seed(42)

# ── Aircraft type pools ───────────────────────────────────────────────────────
NARROW_BODY = ["B737", "A320", "B738", "A321"]
WIDE_BODY   = ["B777", "B787", "A330", "A350", "B747"]

# ── Realistic name pools ──────────────────────────────────────────────────────
FIRST_NAMES = [
    "James","Maria","Wei","Priya","Omar","Sofia","Liam","Aisha",
    "Noah","Elena","Hiroshi","Fatima","Carlos","Zara","Ethan",
    "Amara","Lucas","Mei","Raj","Isabelle","Kwame","Nadia",
    "Arjun","Leila","Marcus","Yuki","Samuel","Hana","Diego","Anna",
]
LAST_NAMES = [
    "Chen","Rodriguez","Patel","Kim","Johnson","Okonkwo","Nakamura",
    "Santos","Mueller","Ahmed","Dubois","Singh","Williams","Garcia",
    "Park","Thompson","Ivanova","Ali","Martínez","Fischer","Brown",
    "Davis","Wilson","Taylor","Anderson","Thomas","Jackson","White",
]


def random_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def random_qualifications(role: str) -> list:
    """
    Captains/FOs are qualified on 1-2 aircraft types.
    FAs have a broader type list (less relevant, but included).
    """
    if role in ("captain", "first_officer"):
        primary = random.choice(NARROW_BODY + WIDE_BODY)
        quals   = [primary]
        # ~40% chance of a second type rating
        if random.random() < 0.4:
            pool = WIDE_BODY if primary in NARROW_BODY else NARROW_BODY
            quals.append(random.choice(pool))
        return list(set(quals))
    else:
        return random.sample(NARROW_BODY + WIDE_BODY, k=random.randint(2, 4))


def make_crew_member(base_airport: str, role: str, index: int) -> dict:
    role_code   = {"captain": "CPT", "first_officer": "FO", "flight_attendant": "FA"}[role]
    employee_id = f"{role_code}{index:05d}"

    # duty state: randomise a realistic current state
    status_weights = {
        "captain":          [("available",0.4),("on_duty",0.15),("rest",0.25),("standby",0.1),("off",0.1)],
        "first_officer":    [("available",0.4),("on_duty",0.15),("rest",0.25),("standby",0.1),("off",0.1)],
        "flight_attendant": [("available",0.35),("on_duty",0.2),("rest",0.2),("standby",0.1),("off",0.15)],
    }[role]
    status = random.choices(
        [s for s,_ in status_weights],
        weights=[w for _,w in status_weights]
    )[0]

    # current duty hours for this duty period
    if status == "on_duty":
        current_duty_hours = round(random.uniform(1.0, 10.0), 1)
    elif status == "rest":
        current_duty_hours = round(random.uniform(10.0, 14.0), 1)  # just finished a long duty
    else:
        current_duty_hours = round(random.uniform(0.0, 3.0), 1)

    # last rest start: within last 24h
    last_rest_start = None
    if status in ("available", "on_duty", "standby"):
        hrs_ago = random.uniform(1, 20)
        last_rest_start = (datetime.utcnow() - timedelta(hours=hrs_ago)).isoformat()

    # 7-day cumulative (FAR 117: max 60h/7 days)
    cumulative_7day = round(random.uniform(current_duty_hours, min(60.0, current_duty_hours + 40)), 1)

    # current location: 80% at base, 20% at another hub
    from ingestion.load_airports import MAJOR_HUBS
    hub_list = sorted(MAJOR_HUBS)
    current_location = base_airport if random.random() < 0.80 else random.choice(hub_list)

    return {
        "id":                     str(uuid.uuid4()),
        "employee_id":            employee_id,
        "name":                   random_name(),
        "role":                   role,
        "base_airport_id":        base_airport,
        "status":                 status,
        "max_duty_hours":         14.0 if role in ("captain","first_officer") else 12.0,
        "current_duty_hours":     current_duty_hours,
        "min_rest_hours":         10.0,
        "last_rest_start":        last_rest_start,
        "cumulative_duty_7day":   cumulative_7day,
        "qualifications_json":    random_qualifications(role),
        "current_location_id":    current_location,
    }


def generate_roster(airports: list, total_crew: int) -> list:
    """
    Distributes crew across airports.
    Ratio: ~1 captain : 1 FO : 3 FAs per station (rough airline norm).
    """
    crew = []
    per_airport = max(1, total_crew // len(airports))

    captain_idx = 1
    fo_idx      = 1
    fa_idx      = 1

    for airport in airports:
        n_captains = max(1, per_airport // 5)
        n_fos      = max(1, per_airport // 5)
        n_fas      = per_airport - n_captains - n_fos

        for _ in range(n_captains):
            crew.append(make_crew_member(airport, "captain",          captain_idx)); captain_idx += 1
        for _ in range(n_fos):
            crew.append(make_crew_member(airport, "first_officer",    fo_idx));      fo_idx      += 1
        for _ in range(n_fas):
            crew.append(make_crew_member(airport, "flight_attendant", fa_idx));      fa_idx      += 1

    log.info(f"Generated {len(crew)} crew members across {len(airports)} airports")
    roles = {}
    for c in crew:
        roles[c["role"]] = roles.get(c["role"], 0) + 1
    for role, count in roles.items():
        log.info(f"  {role}: {count}")
    return crew


def save_to_json(crew: list, path: str = "data/processed/crew_roster.json"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(crew, indent=2))
    log.info(f"Saved crew roster to {p}")


def seed_db(crew: list):
    try:
        from backend.db import get_db_ctx
        from backend.models import CrewMember
        from sqlalchemy.dialects.postgresql import insert
    except ImportError:
        log.warning("Could not import backend. Skipping DB seed.")
        return

    with get_db_ctx() as db:
        existing = {c.employee_id for c in db.query(CrewMember.employee_id).all()}
        new_crew = [
            CrewMember(
                id=c["id"],
                employee_id=c["employee_id"],
                name=c["name"],
                role=c["role"],
                base_airport_id=c["base_airport_id"],
                status=c["status"],
                max_duty_hours=c["max_duty_hours"],
                current_duty_hours=c["current_duty_hours"],
                min_rest_hours=c["min_rest_hours"],
                cumulative_duty_7day=c["cumulative_duty_7day"],
                qualifications_json=c["qualifications_json"],
                current_location_id=c["current_location_id"],
            )
            for c in crew if c["employee_id"] not in existing
        ]
        db.bulk_save_objects(new_crew)
        log.info(f"Inserted {len(new_crew)} crew members into DB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airports", nargs="+",
                        default=["ATL","ORD","LAX","JFK","LHR","DXB","DEL","SIN"],
                        help="Hub airports to distribute crew across")
    parser.add_argument("--count", type=int, default=120,
                        help="Total crew to generate")
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args()

    crew = generate_roster(args.airports, args.count)
    save_to_json(crew)

    if not args.no_db:
        seed_db(crew)
    else:
        log.info("Skipping DB seed (--no-db)")


if __name__ == "__main__":
    main()