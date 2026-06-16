"""
backend/routers/weather.py  +  ml/taf_parser.py (inline)

TAF/METAR weather capacity modeling.

Since we use synthetic data (no live METAR feed), this module:
  1. Parses TAF-format strings into structured forecasts
  2. Scores each condition group for severity
  3. Computes airport capacity reduction (% of normal throughput)
  4. Integrates with cascade model features via get_airport_weather_features()

Capacity reduction model (simplified FAA ATCSCC rules):
  VFR  (ceiling >3000ft, vis >5sm)  → 100% capacity
  MVFR (ceiling 1000-3000, vis 3-5) →  80% capacity
  IFR  (ceiling 500-1000,  vis 1-3) →  55% capacity
  LIFR (ceiling <500,      vis <1)  →  30% capacity  ← Low IFR, LVP mode

Endpoints:
  POST /weather/parse         Parse a TAF string → structured forecast
  POST /weather/capacity      Compute airport capacity from conditions
  GET  /weather/synthetic     Generate synthetic weather for all hub airports
  GET  /weather/impact        Current weather impact on network (synthetic)
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import re, random, math
from datetime import datetime, timezone, timedelta

router = APIRouter()


# ── TAF condition groups ──────────────────────────────────────────────────────

FLIGHT_CATEGORIES = {
    "VFR":  {"ceiling_ft_min": 3000, "vis_sm_min": 5.0,  "capacity_pct": 100},
    "MVFR": {"ceiling_ft_min": 1000, "vis_sm_min": 3.0,  "capacity_pct": 80},
    "IFR":  {"ceiling_ft_min": 500,  "vis_sm_min": 1.0,  "capacity_pct": 55},
    "LIFR": {"ceiling_ft_min": 0,    "vis_sm_min": 0.0,  "capacity_pct": 30},
}

PRECIP_SEVERITY = {
    "TS":   0.30,   # thunderstorm
    "TSRA": 0.35,
    "SN":   0.20,   # snow
    "BLSN": 0.25,   # blowing snow
    "FZRA": 0.30,   # freezing rain
    "RA":   0.05,   # rain
    "DZ":   0.03,   # drizzle
    "FG":   0.15,   # fog
    "BR":   0.05,   # mist
    "HZ":   0.05,   # haze
    "GR":   0.20,   # hail
}


def _classify_flight_category(ceiling_ft: Optional[float], vis_sm: Optional[float]) -> str:
    """Return VFR/MVFR/IFR/LIFR given ceiling (ft) and visibility (statute miles)."""
    if ceiling_ft is None:
        ceiling_ft = 9999
    if vis_sm is None:
        vis_sm = 10.0

    if ceiling_ft < 500 or vis_sm < 1.0:
        return "LIFR"
    elif ceiling_ft < 1000 or vis_sm < 3.0:
        return "IFR"
    elif ceiling_ft < 3000 or vis_sm < 5.0:
        return "MVFR"
    return "VFR"


def _capacity_from_category(category: str, precip_codes: list[str]) -> float:
    """
    Base capacity from flight category, reduced further by precip type.
    Returns fraction 0.0–1.0.
    """
    base = FLIGHT_CATEGORIES.get(category, {}).get("capacity_pct", 100) / 100.0
    penalty = max((PRECIP_SEVERITY.get(p, 0) for p in precip_codes), default=0)
    return max(0.15, base - penalty)


def _parse_ceiling(taf_group: str) -> Optional[float]:
    """Extract lowest ceiling in feet from a TAF group string."""
    # BKN or OVC followed by 3-digit height (hundreds of feet)
    matches = re.findall(r'(?:BKN|OVC)(\d{3})', taf_group)
    if matches:
        return min(int(h) * 100 for h in matches)
    return None


def _parse_visibility(taf_group: str) -> Optional[float]:
    """Extract visibility in statute miles from a TAF group string."""
    # e.g. "9999" (meters, ICAO) or "6SM" or "1/2SM"
    # SM format
    sm = re.search(r'(\d+(?:/\d+)?)\s*SM', taf_group)
    if sm:
        raw = sm.group(1)
        if '/' in raw:
            n, d = raw.split('/')
            return int(n) / int(d)
        return float(raw)
    # ICAO meters (4-digit, e.g. 9999 = 10km+, 0800 = 800m)
    m = re.search(r'\b(\d{4})\b', taf_group)
    if m:
        meters = int(m.group(1))
        return meters / 1609.34   # convert to statute miles
    return None


def _parse_precip(taf_group: str) -> list[str]:
    """Extract precipitation/obstruction codes."""
    codes = []
    for code in PRECIP_SEVERITY:
        if code in taf_group:
            codes.append(code)
    return codes


def parse_taf(taf_string: str, airport_id: str = "UNKN") -> dict:
    """
    Parse a TAF string into a structured forecast dict.

    Example input:
      "TAF KORD 161130Z 1612/1718 27015KT 9999 FEW030
         TEMPO 1615/1620 27025G40KT 3SM TSRA BKN020CB
         FM162000 30012KT 6SM -SN BKN040
         FM170400 28008KT 9999 SCT060"
    """
    groups = []
    lines  = taf_string.strip().replace("\n", " ").split("FM")
    if not lines:
        return {"airport": airport_id, "groups": []}

    # process base + FM groups
    all_chunks = [lines[0]] + [("FM" + l) for l in lines[1:]]

    for chunk in all_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        ceiling = _parse_ceiling(chunk)
        vis     = _parse_visibility(chunk)
        precip  = _parse_precip(chunk)
        cat     = _classify_flight_category(ceiling, vis)
        cap     = _capacity_from_category(cat, precip)

        # wind
        wind = re.search(r'(\d{5}(?:G\d{2,3})?KT)', chunk)

        groups.append({
            "raw":            chunk[:60],
            "flight_category": cat,
            "ceiling_ft":     ceiling,
            "visibility_sm":  vis,
            "precip_codes":   precip,
            "capacity_pct":   round(cap * 100),
            "wind_raw":       wind.group(1) if wind else None,
        })

    worst_cap = min((g["capacity_pct"] for g in groups), default=100)
    worst_cat = min(
        (g["flight_category"] for g in groups),
        key=lambda c: ["VFR","MVFR","IFR","LIFR"].index(c)
    ) if groups else "VFR"

    return {
        "airport":         airport_id,
        "parsed_at":       datetime.now(timezone.utc).isoformat(),
        "groups":          groups,
        "worst_category":  worst_cat,
        "worst_capacity_pct": worst_cap,
        "lvp_active":      worst_cap <= 30,
        "capacity_fraction": worst_cap / 100.0,
    }


def compute_capacity(
    ceiling_ft:    Optional[float],
    vis_sm:        Optional[float],
    precip_codes:  list[str],
    wind_speed_kt: Optional[float] = None,
) -> dict:
    """Direct capacity calculation from conditions (no TAF string needed)."""
    cat = _classify_flight_category(ceiling_ft, vis_sm)
    cap = _capacity_from_category(cat, precip_codes)

    # crosswind component can further reduce capacity
    if wind_speed_kt and wind_speed_kt > 30:
        cap = max(0.15, cap - 0.10)
    if wind_speed_kt and wind_speed_kt > 40:
        cap = max(0.15, cap - 0.15)

    return {
        "flight_category":  cat,
        "capacity_pct":     round(cap * 100),
        "capacity_fraction": cap,
        "lvp_active":       cap <= 0.30,
        "ceiling_ft":       ceiling_ft,
        "visibility_sm":    vis_sm,
        "precip_codes":     precip_codes,
        "wind_speed_kt":    wind_speed_kt,
    }


def get_airport_weather_features(airport_id: str, weather_map: dict) -> dict:
    """
    Return weather feature dict for ML cascade model integration.
    weather_map: {airport_id: capacity_result_dict}
    """
    w = weather_map.get(airport_id, {})
    cat = w.get("flight_category", "VFR")
    cat_score = {"VFR": 0, "MVFR": 1, "IFR": 2, "LIFR": 3}.get(cat, 0)
    return {
        "weather_category_score": cat_score,
        "capacity_fraction":      w.get("capacity_fraction", 1.0),
        "lvp_active":             int(w.get("lvp_active", False)),
        "has_ts":                 int("TS" in w.get("precip_codes", [])),
        "has_snow":               int("SN" in w.get("precip_codes", []) or
                                      "BLSN" in w.get("precip_codes", [])),
    }


# ── Synthetic weather generator for hub airports ──────────────────────────────

HUB_AIRPORTS = [
    "ATL","ORD","DFW","DEN","LAX","JFK","SFO","SEA","MIA","BOS",
    "LAS","PHX","CLT","EWR","MCO","IAH","MSP","DTW","LGA","FLL",
]

SCENARIO_WEIGHTS = {
    "clear":      0.45,
    "mvfr":       0.20,
    "ifr":        0.15,
    "lifr":       0.05,
    "snowstorm":  0.05,
    "thunderstorm": 0.10,
}

def _synthetic_weather_for_airport(airport: str, seed: int = 0) -> dict:
    rng = random.Random(seed + hash(airport) % 1000)
    scenario = rng.choices(
        list(SCENARIO_WEIGHTS.keys()),
        weights=list(SCENARIO_WEIGHTS.values())
    )[0]

    if scenario == "clear":
        ceiling, vis, precip, wind = 8000, 10.0, [], 8
    elif scenario == "mvfr":
        ceiling, vis, precip, wind = rng.randint(1500, 2999), rng.uniform(3, 5), ["BR"], 12
    elif scenario == "ifr":
        ceiling, vis, precip, wind = rng.randint(500, 999), rng.uniform(1, 3), ["FG","RA"], 15
    elif scenario == "lifr":
        ceiling, vis, precip, wind = rng.randint(100, 499), rng.uniform(0.1, 0.9), ["FG"], 10
    elif scenario == "snowstorm":
        ceiling, vis, precip, wind = rng.randint(800, 1500), rng.uniform(0.5, 2), ["SN","BLSN"], rng.randint(20,35)
    else:  # thunderstorm
        ceiling, vis, precip, wind = rng.randint(1000, 2500), rng.uniform(1, 3), ["TS","TSRA"], rng.randint(15,30)

    result = compute_capacity(ceiling, vis, precip, wind)
    result["airport"]  = airport
    result["scenario"] = scenario
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/parse")
def parse_taf_endpoint(body: dict):
    taf_string = body.get("taf")
    airport    = body.get("airport", "UNKN")
    if not taf_string:
        raise HTTPException(400, detail="taf field required")
    return parse_taf(taf_string, airport)


@router.post("/capacity")
def capacity_endpoint(body: dict):
    return compute_capacity(
        ceiling_ft    = body.get("ceiling_ft"),
        vis_sm        = body.get("visibility_sm"),
        precip_codes  = body.get("precip_codes", []),
        wind_speed_kt = body.get("wind_speed_kt"),
    )


@router.get("/synthetic")
def synthetic_weather(seed: int = 42):
    """Generate synthetic weather snapshot for all hub airports."""
    now  = datetime.now(timezone.utc)
    data = {}
    for ap in HUB_AIRPORTS:
        w = _synthetic_weather_for_airport(ap, seed=seed + now.hour)
        data[ap] = w

    # network-level summary
    capacities = [w["capacity_pct"] for w in data.values()]
    lvp_active = [ap for ap, w in data.items() if w["lvp_active"]]
    avg_cap    = round(sum(capacities) / len(capacities), 1)

    return {
        "snapshot_time":   now.isoformat(),
        "airports":        data,
        "network_summary": {
            "avg_capacity_pct": avg_cap,
            "lvp_airports":     lvp_active,
            "restricted_count": sum(1 for c in capacities if c < 80),
            "total_airports":   len(data),
        },
    }


@router.get("/impact")
def weather_impact(seed: int = 42):
    """
    Compute cascade risk uplift from current synthetic weather.
    For each airport in LVP/IFR, estimate additional delay minutes
    and departure rate reduction.
    """
    snap = synthetic_weather(seed=seed)
    airports = snap["airports"]

    impacts = []
    for ap, w in airports.items():
        cap = w["capacity_pct"]
        if cap < 100:
            # estimated additional delay: simplified M/M/1 queue model
            # λ/μ = utilisation. At 55% capacity, delays grow nonlinearly.
            utilisation = min(0.99, 1.0 / (cap / 100.0))
            extra_delay  = round(15 * (utilisation / (1 - min(utilisation, 0.98))), 0)
            extra_delay  = min(extra_delay, 180)   # cap at 3h for display

            impacts.append({
                "airport":          ap,
                "flight_category":  w["flight_category"],
                "capacity_pct":     cap,
                "scenario":         w["scenario"],
                "lvp_active":       w["lvp_active"],
                "extra_delay_est_min": int(extra_delay),
                "precip":           w["precip_codes"],
            })

    impacts.sort(key=lambda x: x["capacity_pct"])

    return {
        "snapshot_time":   snap["snapshot_time"],
        "impacted_airports": impacts,
        "network_summary": snap["network_summary"],
    }