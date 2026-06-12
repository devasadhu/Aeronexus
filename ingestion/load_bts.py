"""
ingestion/load_bts.py

Downloads BTS On-Time Performance data for a subset of carriers/airports,
normalises it into the flights table schema, and optionally seeds the DB.

BTS CSV columns we use:
  YEAR, MONTH, DAY_OF_MONTH, DAY_OF_WEEK, OP_CARRIER, TAIL_NUM,
  ORIGIN, DEST, CRS_DEP_TIME, DEP_TIME, DEP_DELAY, DEP_DEL15,
  CRS_ARR_TIME, ARR_TIME, ARR_DELAY, CANCELLED, CANCELLATION_CODE,
  DIVERTED, CRS_ELAPSED_TIME, ACTUAL_ELAPSED_TIME, DISTANCE

Since BTS bulk download requires registration, we use the public
Kaggle mirror or generate a realistic synthetic substitute.

Usage:
    python -m ingestion.load_bts [--csv path/to/bts.csv] [--sample 5000] [--no-db]
    python -m ingestion.load_bts --synthetic --sample 5000 --no-db
"""

import sys, uuid, random, json, argparse, logging, io
from pathlib import Path
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

random.seed(99)
np.random.seed(99)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Airports we keep — must match our airports table
VALID_AIRPORTS = {
    "ATL","ORD","LAX","DFW","DEN","JFK","SFO","LAS","SEA","MCO",
    "MIA","CLT","PHX","EWR","IAH","BOS","MSP","DTW","FLL","PHL",
    "LHR","CDG","AMS","FRA","DXB","SIN","DEL","BOM","NRT","ICN",
}

CARRIERS = ["AA","UA","DL","WN","B6","AS","NK","F9","HA","G4"]

# Delay cause distributions (minutes) — based on BTS historical patterns
DELAY_DIST = {
    "weather":    {"prob": 0.15, "mu": 45,  "sigma": 30},
    "carrier":    {"prob": 0.35, "mu": 25,  "sigma": 20},
    "nas":        {"prob": 0.25, "mu": 18,  "sigma": 12},   # NAS = ATC/airspace
    "late_acft":  {"prob": 0.20, "mu": 35,  "sigma": 25},
    "security":   {"prob": 0.05, "mu": 10,  "sigma": 5},
}

CANCELLATION_CODES = {
    "A": "carrier", "B": "weather", "C": "atc", "D": "security"
}


# ── Synthetic BTS generator ───────────────────────────────────────────────────

def generate_synthetic_bts(n_flights: int = 5000) -> pd.DataFrame:
    """
    Creates a realistic synthetic BTS-like DataFrame.
    Mirrors real BTS column names for drop-in compatibility.
    """
    airports = sorted(VALID_AIRPORTS)
    rows = []

    base_date = date(2024, 1, 1)

    for i in range(n_flights):
        day_offset = random.randint(0, 364)
        flight_date = base_date + timedelta(days=day_offset)

        carrier  = random.choice(CARRIERS)
        origin   = random.choice(airports)
        dest     = random.choice([a for a in airports if a != origin])

        # scheduled times
        dep_hour = random.choices(range(6, 23), weights=[
            3,4,5,6,8,8,7,6,6,5,5,5,4,4,5,5,4], k=1)[0]
        dep_min  = random.choice([0, 15, 30, 45])
        crs_dep  = dep_hour * 100 + dep_min

        dist_map = {(o, d): random.randint(300, 8000)
                    for o in airports for d in airports}
        distance = dist_map.get((origin, dest), 1500)
        elapsed  = int(distance / 13.5 + 30)   # rough minutes
        arr_hour = (dep_hour * 60 + dep_min + elapsed) // 60
        arr_min  = (dep_hour * 60 + dep_min + elapsed) % 60
        crs_arr  = (arr_hour % 24) * 100 + arr_min

        # disruption
        cancelled = random.random() < 0.02
        diverted  = (not cancelled) and random.random() < 0.005

        if cancelled:
            cancel_code = random.choices(["A","B","C","D"], weights=[40,35,20,5])[0]
            dep_delay = np.nan
            arr_delay = np.nan
            actual_elapsed = np.nan
        else:
            # delay logic
            delay = 0.0
            cause = None
            for cause_name, params in DELAY_DIST.items():
                if random.random() < params["prob"]:
                    d = max(0, np.random.normal(params["mu"], params["sigma"]))
                    if d > delay:
                        delay = d
                        cause = cause_name
            dep_delay = round(delay, 1)
            arr_delay = round(delay - random.uniform(0, min(delay, 10)), 1)
            actual_elapsed = elapsed + int(dep_delay)
            cancel_code = np.nan

        rows.append({
            "YEAR":              flight_date.year,
            "MONTH":             flight_date.month,
            "DAY_OF_MONTH":      flight_date.day,
            "DAY_OF_WEEK":       flight_date.isoweekday(),
            "FL_DATE":           flight_date.isoformat(),
            "OP_CARRIER":        carrier,
            "TAIL_NUM":          f"N{random.randint(10000,99999)}{random.choice('ABCD')}",
            "OP_CARRIER_FL_NUM": random.randint(100, 5999),
            "ORIGIN":            origin,
            "DEST":              dest,
            "CRS_DEP_TIME":      crs_dep,
            "DEP_TIME":          crs_dep + int(dep_delay) if not cancelled else np.nan,
            "DEP_DELAY":         dep_delay,
            "DEP_DEL15":         1 if (not np.isnan(dep_delay) if not cancelled else False) and dep_delay >= 15 else 0,
            "CRS_ARR_TIME":      crs_arr,
            "ARR_TIME":          crs_arr + int(arr_delay) if not cancelled else np.nan,
            "ARR_DELAY":         arr_delay,
            "CANCELLED":         1 if cancelled else 0,
            "CANCELLATION_CODE": cancel_code,
            "DIVERTED":          1 if diverted else 0,
            "CRS_ELAPSED_TIME":  elapsed,
            "ACTUAL_ELAPSED_TIME": actual_elapsed,
            "DISTANCE":          distance,
        })

    df = pd.DataFrame(rows)
    log.info(f"Generated synthetic BTS: {len(df)} flights")
    log.info(f"  Cancelled: {df['CANCELLED'].sum()} ({df['CANCELLED'].mean()*100:.1f}%)")
    log.info(f"  Diverted:  {df['DIVERTED'].sum()}")
    log.info(f"  Delayed≥15: {df['DEP_DEL15'].sum()} ({df['DEP_DEL15'].mean()*100:.1f}%)")
    return df


# ── BTS CSV parser ────────────────────────────────────────────────────────────

def parse_bts_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]

    # keep only airports in our network
    df = df[df["ORIGIN"].isin(VALID_AIRPORTS) & df["DEST"].isin(VALID_AIRPORTS)]
    log.info(f"Loaded {len(df)} BTS rows after airport filter")
    return df


# ── Time helpers ──────────────────────────────────────────────────────────────

def hhmm_to_time(fl_date_str: str, hhmm) -> datetime | None:
    """Convert BTS HHMM integer + date string to datetime."""
    if pd.isna(hhmm):
        return None
    try:
        hhmm_int = int(hhmm)
        h = hhmm_int // 100
        m = hhmm_int % 100
        if h >= 24:          # overnight adjustment
            h = h % 24
        d = datetime.fromisoformat(fl_date_str)
        return d.replace(hour=h, minute=m, second=0, microsecond=0)
    except Exception:
        return None


# ── Normalise to Flight dicts ─────────────────────────────────────────────────

def normalise_to_flights(df: pd.DataFrame) -> list:
    flights = []

    for _, row in df.iterrows():
        fl_date = str(row.get("FL_DATE", f"{int(row['YEAR'])}-{int(row['MONTH']):02d}-{int(row['DAY_OF_MONTH']):02d}"))

        sched_dep = hhmm_to_time(fl_date, row.get("CRS_DEP_TIME"))
        sched_arr = hhmm_to_time(fl_date, row.get("CRS_ARR_TIME"))
        if sched_dep is None or sched_arr is None:
            continue

        # handle overnight arrivals
        if sched_arr < sched_dep:
            sched_arr += timedelta(days=1)

        delay_min = 0
        actual_dep = None
        actual_arr = None
        status = "scheduled"

        cancelled = int(row.get("CANCELLED", 0)) == 1
        diverted  = int(row.get("DIVERTED",  0)) == 1

        if cancelled:
            status = "cancelled"
            cancel_code = str(row.get("CANCELLATION_CODE", "A"))
            cancel_reason = CANCELLATION_CODES.get(cancel_code, "unknown")
        else:
            dep_delay = row.get("DEP_DELAY", 0)
            delay_min = int(dep_delay) if not pd.isna(dep_delay) else 0
            actual_dep = hhmm_to_time(fl_date, row.get("DEP_TIME"))
            actual_arr = hhmm_to_time(fl_date, row.get("ARR_TIME"))

            if actual_arr and actual_dep and actual_arr < actual_dep:
                actual_arr += timedelta(days=1)

            if diverted:
                status = "diverted"
            elif delay_min >= 15:
                status = "delayed"
            else:
                status = "arrived"

            cancel_reason = None

        flights.append({
            "id":                  str(uuid.uuid4()),
            "flight_number":       f"{row['OP_CARRIER']}{int(row.get('OP_CARRIER_FL_NUM', 0))}",
            "airline_code":        str(row["OP_CARRIER"]),
            "origin_id":           str(row["ORIGIN"]),
            "destination_id":      str(row["DEST"]),
            "aircraft_id":         None,
            "scheduled_departure": sched_dep,
            "scheduled_arrival":   sched_arr,
            "actual_departure":    actual_dep,
            "actual_arrival":      actual_arr,
            "status":              status,
            "delay_minutes":       max(0, delay_min),
            "cancellation_reason": cancel_reason if cancelled else None,
            "tail_number":         str(row.get("TAIL_NUM", "")) or None,
            "capacity":            random.randint(120, 200),
            "booked_seats":        0,
            "bts_carrier":         str(row["OP_CARRIER"]),
            "bts_origin":          str(row["ORIGIN"]),
            "bts_dest":            str(row["DEST"]),
            "year":                int(row["YEAR"]),
            "month":               int(row["MONTH"]),
            "day_of_week":         int(row.get("DAY_OF_WEEK", 1)),
        })

    log.info(f"Normalised {len(flights)} flight records")
    counts = {}
    for f in flights:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    for s, c in sorted(counts.items()):
        log.info(f"  {s}: {c}")
    return flights


# ── Identify disruptions from flight data ─────────────────────────────────────

def identify_disruptions(flights: list, delay_threshold_min: int = 60) -> list:
    """
    Flag flights as disruption roots based on:
    - Delay >= threshold
    - Cancellation
    - Diversion
    Returns list of disruption seed dicts (used by cascade model).
    """
    disruptions = []
    for f in flights:
        if f["status"] == "cancelled":
            disruptions.append({
                "flight_id":      f["id"],
                "flight_number":  f["flight_number"],
                "type":           "carrier" if f.get("cancellation_reason") == "carrier" else "weather",
                "delay_minutes":  0,
                "origin":         f["origin_id"],
                "destination":    f["destination_id"],
                "departure_time": f["scheduled_departure"].isoformat()
                                  if hasattr(f["scheduled_departure"], "isoformat")
                                  else str(f["scheduled_departure"]),
            })
        elif f["status"] == "diverted":
            disruptions.append({
                "flight_id":      f["id"],
                "flight_number":  f["flight_number"],
                "type":           "weather",
                "delay_minutes":  f["delay_minutes"],
                "origin":         f["origin_id"],
                "destination":    f["destination_id"],
                "departure_time": f["scheduled_departure"].isoformat()
                                  if hasattr(f["scheduled_departure"], "isoformat")
                                  else str(f["scheduled_departure"]),
            })
        elif f["delay_minutes"] >= delay_threshold_min:
            disruptions.append({
                "flight_id":      f["id"],
                "flight_number":  f["flight_number"],
                "type":           "unknown",
                "delay_minutes":  f["delay_minutes"],
                "origin":         f["origin_id"],
                "destination":    f["destination_id"],
                "departure_time": f["scheduled_departure"].isoformat()
                                  if hasattr(f["scheduled_departure"], "isoformat")
                                  else str(f["scheduled_departure"]),
            })

    log.info(f"Identified {len(disruptions)} disruption events from {len(flights)} flights")
    return disruptions


# ── Save ──────────────────────────────────────────────────────────────────────

def save_to_json(flights: list, disruptions: list):
    processed = DATA_DIR / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # serialise datetimes
    def serial(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    (processed / "flights.json").write_text(
        json.dumps(flights, indent=2, default=serial))
    (processed / "disruptions_seed.json").write_text(
        json.dumps(disruptions, indent=2, default=serial))

    log.info(f"Saved {len(flights)} flights → data/processed/flights.json")
    log.info(f"Saved {len(disruptions)} disruptions → data/processed/disruptions_seed.json")


def seed_db(flights: list):
    try:
        from backend.db import get_db_ctx
        from backend.models import Flight
    except ImportError:
        log.warning("Could not import backend. Skipping DB seed.")
        return

    with get_db_ctx() as db:
        existing = {f.flight_number + str(f.scheduled_departure)
                    for f in db.query(Flight.flight_number, Flight.scheduled_departure).all()}
        new_flights = []
        for f in flights:
            key = f["flight_number"] + str(f["scheduled_departure"])
            if key not in existing:
                new_flights.append(Flight(**{k: v for k, v in f.items()
                                             if k not in ("bts_carrier","bts_origin","bts_dest")}))
        db.bulk_save_objects(new_flights)
        log.info(f"Inserted {len(new_flights)} flights into DB")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default=None,  help="Path to BTS CSV file")
    parser.add_argument("--synthetic",  action="store_true", help="Generate synthetic data")
    parser.add_argument("--sample",     type=int,  default=5000)
    parser.add_argument("--delay-threshold", type=int, default=60)
    parser.add_argument("--no-db",      action="store_true")
    args = parser.parse_args()

    if args.csv and Path(args.csv).exists():
        df = parse_bts_csv(args.csv)
    else:
        if not args.synthetic:
            log.info("No CSV provided, falling back to --synthetic mode")
        df = generate_synthetic_bts(args.sample)

    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)
        log.info(f"Sampled down to {len(df)} rows")

    flights     = normalise_to_flights(df)
    disruptions = identify_disruptions(flights, args.delay_threshold)
    save_to_json(flights, disruptions)

    if not args.no_db:
        seed_db(flights)
    else:
        log.info("Skipping DB seed (--no-db)")


if __name__ == "__main__":
    main()