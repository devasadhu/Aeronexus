"""
synthetic/gen_passengers.py

Generates synthetic passenger booking records (PNRs) with:
- Single-leg and multi-leg itineraries
- Realistic connection times
- Fare class distribution
- Pre-computed misconnect risk based on connection time

Requires gen_aircraft.py output (fleet) and the flight graph to be built.
Can also operate in standalone mode with a hardcoded flight list.

Usage:
    python -m synthetic.gen_passengers [--flights-json path] [--pax-count 500] [--no-db]
"""

import sys, uuid, random, json, argparse, logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

random.seed(13)

FIRST_NAMES = [
    "Alice","Bob","Charlie","Diana","Eve","Frank","Grace","Hank",
    "Iris","Jack","Karen","Leo","Mona","Nate","Olivia","Paul",
    "Quinn","Rosa","Sam","Tara","Uma","Victor","Wendy","Xander",
    "Yara","Zach","Amir","Beatrice","Cyrus","Daphne",
]
LAST_NAMES = [
    "Smith","Jones","Wang","Patel","Brown","Garcia","Müller","Tanaka",
    "Okafor","Ivanov","Dubois","Rossi","Kim","Silva","Andersen",
    "Nguyen","Hassan","Kowalski","Yamamoto","Fernandez",
]

FARE_CLASS_WEIGHTS = [
    ("first",    0.05),
    ("business", 0.15),
    ("economy",  0.80),
]

# Minimum connection times by airport type (minutes)
MCT = {
    "hub":   45,
    "spoke": 30,
}

HUB_AIRPORTS = {
    "ATL","ORD","LAX","DFW","DEN","JFK","SFO","LHR","CDG","AMS",
    "FRA","DXB","SIN","HKG","NRT","PEK","DEL","BOM",
}


def random_pnr() -> str:
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(random.choices(chars, k=6))


def misconnect_probability(connection_time_min: int, is_hub: bool) -> float:
    """
    Sigmoid-like function: short connections = high risk.
    Hub airports have slightly lower risk (better infrastructure).
    """
    mct = MCT["hub"] if is_hub else MCT["spoke"]
    buffer = connection_time_min - mct
    if buffer <= 0:
        return 0.95
    elif buffer <= 15:
        return 0.70
    elif buffer <= 30:
        return 0.35
    elif buffer <= 60:
        return 0.10
    else:
        return 0.02


def make_passenger() -> dict:
    fare_class = random.choices(
        [f for f, _ in FARE_CLASS_WEIGHTS],
        weights=[w for _, w in FARE_CLASS_WEIGHTS]
    )[0]

    return {
        "id":             str(uuid.uuid4()),
        "pnr":            random_pnr(),
        "name":           f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
        "fare_class":     fare_class,
        "frequent_flyer": fare_class in ("first", "business") and random.random() < 0.6,
        "contact_email":  None,
    }


def make_itinerary(passenger: dict, flights: list, leg_count: int = 1) -> list:
    """
    Assign passenger to 1 or 2 flight legs.
    Returns list of itinerary dicts.
    """
    if not flights:
        return []

    itins = []
    chosen_flights = random.sample(flights, min(leg_count, len(flights)))

    for leg_num, flight in enumerate(chosen_flights, start=1):
        connection_time_min = None
        misconnect_risk = 0.0

        if leg_num > 1:
            # simulate connection time based on previous leg's arrival
            # in a full system this would be computed from actual times
            connection_time_min = random.randint(20, 120)
            is_hub = flight.get("origin_id", "") in HUB_AIRPORTS
            misconnect_risk = misconnect_probability(connection_time_min, is_hub)

        itins.append({
            "id":                   str(uuid.uuid4()),
            "passenger_id":         passenger["id"],
            "flight_id":            flight["id"],
            "leg_number":           leg_num,
            "connection_time_min":  connection_time_min,
            "misconnect_risk":      round(misconnect_risk, 3),
            "rebooked":             False,
            "rebooked_flight_id":   None,
        })

    return itins


def generate_passengers(flights: list, pax_count: int) -> tuple:
    """
    Returns (passengers, itineraries).
    ~70% single-leg, ~30% two-leg itineraries.
    """
    passengers  = []
    itineraries = []

    for _ in range(pax_count):
        pax = make_passenger()
        passengers.append(pax)

        leg_count = 2 if random.random() < 0.30 else 1
        itins = make_itinerary(pax, flights, leg_count=leg_count)
        itineraries.extend(itins)

    # stats
    single_leg = sum(1 for p in passengers
                     if sum(1 for i in itineraries if i["passenger_id"] == p["id"]) == 1)
    multi_leg  = pax_count - single_leg
    high_risk  = sum(1 for i in itineraries if i["misconnect_risk"] > 0.35)
    ff_count   = sum(1 for p in passengers if p["frequent_flyer"])

    log.info(f"Generated {pax_count} passengers, {len(itineraries)} itinerary legs")
    log.info(f"  Single-leg: {single_leg}  Multi-leg: {multi_leg}")
    log.info(f"  High misconnect risk (>35%): {high_risk} legs")
    log.info(f"  Frequent flyers: {ff_count}")

    return passengers, itineraries


def make_stub_flights(airports: list, n: int = 50) -> list:
    """
    Standalone mode: generate minimal flight stubs when no flight DB exists yet.
    These are replaced by real flights once load_bts.py runs.
    """
    flights = []
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    for i in range(n):
        origin = random.choice(airports)
        dest   = random.choice([a for a in airports if a != origin])
        dep    = now + timedelta(hours=random.randint(1, 24))
        arr    = dep + timedelta(minutes=random.randint(90, 480))

        flights.append({
            "id":           str(uuid.uuid4()),
            "flight_number": f"AX{100 + i}",
            "origin_id":    origin,
            "destination_id": dest,
            "scheduled_departure": dep.isoformat(),
            "scheduled_arrival":   arr.isoformat(),
        })

    log.info(f"Created {len(flights)} stub flights for passenger assignment")
    return flights


def save_to_json(passengers: list, itineraries: list,
                  pax_path: str = "data/processed/passengers.json",
                  itin_path: str = "data/processed/itineraries.json"):
    for data, path in [(passengers, pax_path), (itineraries, itin_path)]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))
        log.info(f"Saved {len(data)} records to {p}")


def seed_db(passengers: list, itineraries: list):
    try:
        from backend.db import get_db_ctx
        from backend.models import Passenger, PassengerItinerary
    except ImportError:
        log.warning("Could not import backend. Skipping DB seed.")
        return

    with get_db_ctx() as db:
        existing_pnrs = {p.pnr for p in db.query(Passenger.pnr).all()}
        new_pax = [
            Passenger(
                id=p["id"], pnr=p["pnr"], name=p["name"],
                fare_class=p["fare_class"], frequent_flyer=p["frequent_flyer"],
            )
            for p in passengers if p["pnr"] not in existing_pnrs
        ]
        db.bulk_save_objects(new_pax)
        db.flush()

        new_itins = [
            PassengerItinerary(
                id=i["id"],
                passenger_id=i["passenger_id"],
                flight_id=i["flight_id"],
                leg_number=i["leg_number"],
                connection_time_min=i["connection_time_min"],
                misconnect_risk=i["misconnect_risk"],
                rebooked=i["rebooked"],
            )
            for i in itineraries
        ]
        db.bulk_save_objects(new_itins)
        log.info(f"Inserted {len(new_pax)} passengers, {len(new_itins)} itinerary legs into DB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flights-json", default=None,
                        help="Path to flights JSON. If omitted, stub flights are generated.")
    parser.add_argument("--airports", nargs="+",
                        default=["ATL","ORD","LAX","JFK","LHR","DXB","DEL","SIN"])
    parser.add_argument("--pax-count", type=int, default=500)
    parser.add_argument("--no-db", action="store_true")
    args = parser.parse_args()

    if args.flights_json and Path(args.flights_json).exists():
        flights = json.loads(Path(args.flights_json).read_text())
        log.info(f"Loaded {len(flights)} flights from {args.flights_json}")
    else:
        log.info("No flights JSON provided. Generating stub flights.")
        flights = make_stub_flights(args.airports, n=80)

    passengers, itineraries = generate_passengers(flights, args.pax_count)
    save_to_json(passengers, itineraries)

    if not args.no_db:
        seed_db(passengers, itineraries)
    else:
        log.info("Skipping DB seed (--no-db)")


if __name__ == "__main__":
    main()