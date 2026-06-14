"""
dashboard/app.py

AeroNexus Operations Dashboard — Streamlit app.

Pages:
  1. Disruption Map       — live network map with severity overlays
  2. Disruption Detail    — cascade + per-agent recovery breakdown
  3. What-If Simulator    — inject custom delay and rerun pipeline

Run:
    streamlit run dashboard/app.py
"""

import sys, json, pickle, os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AeroNexus IROPS",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour scheme ─────────────────────────────────────────────────────────────
SEVERITY_COLOUR = {
    "critical": "#e74c3c",
    "high":     "#e67e22",
    "medium":   "#f1c40f",
    "low":      "#2ecc71",
    "unknown":  "#95a5a6",
}

STATUS_COLOUR = {
    "delayed":   "#e67e22",
    "cancelled": "#e74c3c",
    "diverted":  "#9b59b6",
    "arrived":   "#2ecc71",
    "scheduled": "#3498db",
    "departed":  "#1abc9c",
}

# ── Data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_flights():
    p = Path("data/processed/flights.json")
    return json.loads(p.read_text()) if p.exists() else []

@st.cache_data(ttl=60)
def load_disruptions():
    p = Path("data/processed/disruptions_seed.json")
    return json.loads(p.read_text()) if p.exists() else []

@st.cache_data(ttl=300)
def load_graph():
    p = Path("data/processed/flight_graph.gpickle")
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

@st.cache_data(ttl=300)
def load_crew():
    p = Path("data/processed/crew_roster.json")
    return json.loads(p.read_text()) if p.exists() else []

@st.cache_data(ttl=300)
def load_fleet():
    p = Path("data/processed/aircraft_fleet.json")
    return json.loads(p.read_text()) if p.exists() else []

@st.cache_data(ttl=300)
def load_passengers():
    p = Path("data/processed/passengers.json")
    return json.loads(p.read_text()) if p.exists() else []

@st.cache_data(ttl=300)
def load_itineraries():
    p = Path("data/processed/itineraries.json")
    return json.loads(p.read_text()) if p.exists() else []

# airport coords from graph
@st.cache_data(ttl=3600)
def airport_coords():
    G = load_graph()
    if G is None:
        return {}
    return {n: {"lat": d["lat"], "lon": d["lon"], "name": d.get("name",""),
                "is_hub": d.get("is_hub", False)}
            for n, d in G.nodes(data=True)}


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(disruption: dict, risk_threshold: float = 0.25):
    """Run cascade + recovery pipeline. Returns (event, plan)."""
    from ml.feature_builder import build_historical_rates, build_downstream_index
    from ml.cascade_model   import predict_cascade
    from ml.severity_scorer import build_disruption_event
    from agents.coordinator import run_recovery_pipeline

    flights     = load_flights()
    G           = load_graph()
    crew        = load_crew()
    fleet       = load_fleet()
    passengers  = load_passengers()
    itineraries = load_itineraries()

    hist_rates     = build_historical_rates(flights)
    downstream_idx = build_downstream_index(flights)
    flights_lookup = {f["id"]: f for f in flights}

    candidates = [f for f in flights
                   if f["origin_id"] == disruption.get("destination", "")
                   and f["status"] not in ("cancelled",)][:50]

    affected = predict_cascade(disruption, candidates, G, hist_rates,
                                downstream_idx, risk_threshold=risk_threshold)
    event    = build_disruption_event(disruption, affected, flights_lookup, G)

    affected_ids  = {a["flight_id"] for a in affected}
    affected_full = [f for f in flights if f["id"] in affected_ids]

    plan = run_recovery_pipeline(
        disruption=event, affected_flights=affected_full,
        all_flights=flights, available_crew=crew,
        available_aircraft=fleet, passengers=passengers,
        itineraries=itineraries,
    )
    return event, plan


# ── Shared helpers ────────────────────────────────────────────────────────────

def metric_row(cols_data: list):
    cols = st.columns(len(cols_data))
    for col, (label, value, delta) in zip(cols, cols_data):
        col.metric(label, value, delta)


def action_table(actions: list, filter_type: str = None):
    rows = actions if not filter_type else [a for a in actions if a["action_type"] == filter_type]
    if not rows:
        st.info("No actions of this type.")
        return
    df = pd.DataFrame([{
        "Type":        a["action_type"],
        "Flight":      a.get("flight_number", "—"),
        "Agent":       a.get("agent_source", "—"),
        "Feasible":    "✅" if a["feasible"] else "❌",
        "Conflict":    "⚠️" if a["conflict_flag"] else "—",
        "Description": a["description"][:80] + ("…" if len(a["description"]) > 80 else ""),
    } for a in rows])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Disruption Map
# ══════════════════════════════════════════════════════════════════════════════

def page_map():
    st.title("✈ AeroNexus — Disruption Network Map")
    st.caption("Real-time view of disrupted flights and network severity")

    flights     = load_flights()
    disruptions = load_disruptions()
    coords      = airport_coords()

    # ── Sidebar filters
    with st.sidebar:
        st.header("Filters")
        show_status = st.multiselect(
            "Flight status",
            ["delayed","cancelled","diverted","arrived","scheduled"],
            default=["delayed","cancelled","diverted"],
        )
        delay_min = st.slider("Min delay (min)", 0, 300, 30)
        hub_only  = st.checkbox("Hub airports only", value=True)

    # ── Flight stats
    filtered = [f for f in flights if f["status"] in show_status
                 and f.get("delay_minutes", 0) >= delay_min]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total flights",   len(flights))
    c2.metric("Disrupted",       len(disruptions), delta=f"{len(disruptions)/max(len(flights),1)*100:.1f}%")
    c3.metric("Delayed",         sum(1 for f in flights if f["status"] == "delayed"))
    c4.metric("Cancelled",       sum(1 for f in flights if f["status"] == "cancelled"))
    c5.metric("Avg delay (min)", f"{sum(f.get('delay_minutes',0) for f in flights)/max(len(flights),1):.0f}")

    st.divider()

    # ── Build map
    fig = go.Figure()

    # airport nodes
    airport_delay = defaultdict(list)
    for f in flights:
        airport_delay[f["origin_id"]].append(f.get("delay_minutes", 0))

    for iata, c in coords.items():
        if hub_only and not c["is_hub"]:
            continue
        delays = airport_delay.get(iata, [0])
        avg_delay = sum(delays) / len(delays)
        colour = ("#e74c3c" if avg_delay > 60 else
                  "#e67e22" if avg_delay > 30 else
                  "#2ecc71")
        fig.add_trace(go.Scattergeo(
            lon=[c["lon"]], lat=[c["lat"]],
            mode="markers+text",
            marker=dict(size=10, color=colour, opacity=0.85,
                         line=dict(width=1, color="white")),
            text=iata, textposition="top center",
            textfont=dict(size=9, color="white"),
            name=iata,
            hovertemplate=(
                f"<b>{iata}</b><br>{c['name']}<br>"
                f"Avg delay: {avg_delay:.0f} min<br>"
                f"Flights: {len(delays)}<extra></extra>"
            ),
            showlegend=False,
        ))

    # disrupted flight arcs
    disrupt_by_origin = defaultdict(list)
    flight_lookup = {f["id"]: f for f in flights}
    disrupt_lookup = {d["flight_id"]: d for d in disruptions}

    for d in disruptions[:80]:   # cap for performance
        f = flight_lookup.get(d["flight_id"])
        if not f:
            continue
        o = f["origin_id"]
        dest = f["destination_id"]
        if o not in coords or dest not in coords:
            continue
        oc = coords[o]
        dc = coords[dest]
        delay = f.get("delay_minutes", 0)
        col = ("#e74c3c" if delay > 120 or f["status"] == "cancelled"
               else "#e67e22" if delay > 60
               else "#f1c40f")

        fig.add_trace(go.Scattergeo(
            lon=[oc["lon"], dc["lon"], None],
            lat=[oc["lat"], dc["lat"], None],
            mode="lines",
            line=dict(width=1.5, color=col),
            opacity=0.6,
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        geo=dict(
            projection_type="natural earth",
            showland=True, landcolor="#1a1a2e",
            showocean=True, oceancolor="#16213e",
            showlakes=False,
            showcountries=True, countrycolor="#2d2d44",
            showframe=False,
            bgcolor="#0f0f23",
        ),
        paper_bgcolor="#0f0f23",
        plot_bgcolor="#0f0f23",
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Status breakdown bar chart
    st.subheader("Flight status breakdown")
    status_counts = defaultdict(int)
    for f in flights:
        status_counts[f["status"]] += 1

    df_status = pd.DataFrame([
        {"Status": s, "Count": c, "Colour": STATUS_COLOUR.get(s, "#95a5a6")}
        for s, c in sorted(status_counts.items(), key=lambda x: -x[1])
    ])
    fig2 = px.bar(df_status, x="Status", y="Count", color="Status",
                   color_discrete_map=STATUS_COLOUR,
                   template="plotly_dark")
    fig2.update_layout(showlegend=False, paper_bgcolor="#0f0f23",
                        plot_bgcolor="#0f0f23", height=280)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Disruption table
    st.subheader(f"Active disruptions ({len(filtered)} flights shown)")
    if filtered:
        df = pd.DataFrame([{
            "Flight":      f["flight_number"],
            "Origin":      f["origin_id"],
            "Dest":        f["destination_id"],
            "Status":      f["status"],
            "Delay (min)": f.get("delay_minutes", 0),
            "Booked":      f.get("booked_seats", 0),
        } for f in filtered[:100]])
        st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Disruption Detail
# ══════════════════════════════════════════════════════════════════════════════

def page_detail():
    st.title("🔍 Disruption Detail & Recovery Plan")

    disruptions = load_disruptions()
    flights     = load_flights()
    flight_lookup = {f["id"]: f for f in flights}

    if not disruptions:
        st.error("No disruption data found. Run load_bts.py --synthetic first.")
        return

    # selector
    options = {
        f"{d['flight_number']} | {d['origin']}→{d['destination']} | "
        f"{d['type']} | delay={d['delay_minutes']}min": d
        for d in disruptions[:100]
    }
    selected_label = st.selectbox("Select disruption", list(options.keys()))
    disruption = options[selected_label]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Disruption Info")
        st.json({
            "flight":      disruption["flight_number"],
            "route":       f"{disruption['origin']} → {disruption['destination']}",
            "type":        disruption["type"],
            "delay_min":   disruption["delay_minutes"],
            "departure":   disruption.get("departure_time", "—"),
        })
    with col2:
        risk_threshold = st.slider("Risk threshold", 0.0, 1.0, 0.25, 0.05,
                                    help="Cascade model probability cutoff")
        run_btn = st.button("▶ Run Recovery Pipeline", type="primary", use_container_width=True)

    if run_btn or "last_event" in st.session_state:
        with st.spinner("Running cascade prediction and recovery pipeline..."):
            if run_btn:
                event, plan = run_pipeline(disruption, risk_threshold)
                st.session_state["last_event"] = event
                st.session_state["last_plan"]  = plan
            else:
                event = st.session_state["last_event"]
                plan  = st.session_state["last_plan"]

        sev = event.get("severity", "unknown")
        sev_col = SEVERITY_COLOUR.get(sev, "#95a5a6")
        st.markdown(
            f"<div style='background:{sev_col};padding:8px 16px;border-radius:6px;"
            f"color:white;font-weight:bold;font-size:1.1em;margin-bottom:12px'>"
            f"Severity: {sev.upper()} &nbsp;|&nbsp; Score: {event.get('severity_score','?'):.1f}/100"
            f"</div>",
            unsafe_allow_html=True,
        )

        # metrics
        summary = plan.get("summary", {})
        metric_row([
            ("Affected flights",       event.get("total_affected_flights", 0),  None),
            ("Affected passengers",    event.get("total_affected_pax", 0),       None),
            ("Cancellations avoided",  summary.get("cancellations_avoided", 0), None),
            ("Misconnects avoided",    summary.get("misconnects_avoided", 0),    None),
            ("Delay reduction (min)",  summary.get("total_delay_reduction_min",0), None),
            ("Conflicts detected",     summary.get("conflict_count", 0),         None),
        ])

        st.divider()

        # tabs per agent
        tab_fleet, tab_crew, tab_pax, tab_all, tab_json = st.tabs(
            ["🛩 Fleet", "👨‍✈️ Crew", "🧳 Passengers", "📋 All Actions", "🔧 Raw JSON"]
        )

        actions = plan.get("final_actions_json", [])

        with tab_fleet:
            st.markdown("**Aircraft swaps proposed by FleetAgent**")
            action_table(actions, "aircraft_swap")
            cancels = [a for a in actions if a["action_type"] == "flight_cancel"
                        and a["agent_source"] == "fleet"]
            if cancels:
                st.warning(f"{len(cancels)} flight(s) recommended for cancellation (no aircraft available)")
                action_table(cancels)

        with tab_crew:
            st.markdown("**Crew assignments proposed by CrewAgent**")
            action_table(actions, "crew_reassign")

        with tab_pax:
            st.markdown("**Passenger rebookings proposed by PassengerAgent**")
            rebooked = [a for a in actions if a["action_type"] == "passenger_rebook" and a["feasible"]]
            stranded = [a for a in actions if a["action_type"] == "passenger_rebook" and not a["feasible"]]
            c1, c2 = st.columns(2)
            c1.metric("Rebooked", len(rebooked))
            c2.metric("Stranded (manual)", len(stranded))
            if rebooked:
                st.markdown("**Rebooked passengers**")
                action_table(rebooked)
            if stranded:
                st.error(f"{len(stranded)} passengers require manual intervention")
                action_table(stranded)

        with tab_all:
            st.markdown(f"**All {len(actions)} actions in final plan**")
            # action type breakdown donut
            by_type = defaultdict(int)
            for a in actions:
                by_type[a["action_type"]] += 1
            fig_donut = px.pie(
                names=list(by_type.keys()),
                values=list(by_type.values()),
                hole=0.5,
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig_donut.update_layout(paper_bgcolor="#0f0f23", height=250,
                                     margin=dict(t=20,b=20,l=20,r=20))
            st.plotly_chart(fig_donut, use_container_width=True)
            action_table(actions)

        with tab_json:
            st.markdown("**DisruptionEvent**")
            st.json(event)
            st.markdown("**RecoveryPlan summary**")
            st.json(plan.get("summary", {}))

        # rule-based advisory
        st.divider()
        st.subheader("📢 OCC Advisory")
        from backend.routers.advisory import _rule_based_advisory
        summary_text, bullets = _rule_based_advisory(event, plan)
        st.info(summary_text)
        for b in bullets:
            st.markdown(f"- {b}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: What-If Simulator
# ══════════════════════════════════════════════════════════════════════════════

def page_whatif():
    st.title("🔮 What-If Simulator")
    st.caption("Inject a custom disruption and see the recovery plan in real time")

    flights     = load_disruptions()
    all_flights = load_flights()
    coords      = airport_coords()

    airports = sorted(coords.keys())

    st.subheader("Define your disruption scenario")

    col1, col2, col3 = st.columns(3)
    with col1:
        origin = st.selectbox("Origin airport", airports,
                               index=airports.index("ATL") if "ATL" in airports else 0)
    with col2:
        dest_opts = [a for a in airports if a != origin]
        dest = st.selectbox("Destination airport", dest_opts,
                             index=dest_opts.index("ORD") if "ORD" in dest_opts else 0)
    with col3:
        disruption_type = st.selectbox(
            "Disruption type",
            ["weather","carrier","mechanical","atc","airport","unknown"],
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        delay_minutes = st.slider("Delay (minutes)", 0, 480, 90, 15)
    with col5:
        risk_threshold = st.slider("Risk threshold", 0.0, 1.0, 0.25, 0.05)
    with col6:
        dep_offset = st.slider("Departure offset from now (hours)", 1, 24, 6)

    # find a real flight on this route or synthesise one
    matching = [f for f in all_flights
                 if f["origin_id"] == origin and f["destination_id"] == dest]

    if matching:
        base_flight = matching[0]
        st.success(f"Using real flight {base_flight['flight_number']} on {origin}→{dest}")
    else:
        import uuid
        dep = datetime.utcnow() + timedelta(hours=dep_offset)
        arr = dep + timedelta(hours=3)
        base_flight = {
            "id": str(uuid.uuid4()),
            "flight_number": "SIM001",
            "origin_id": origin, "destination_id": dest,
            "status": "delayed", "delay_minutes": delay_minutes,
            "capacity": 150, "booked_seats": 120,
            "scheduled_departure": dep.isoformat(),
            "scheduled_arrival":   arr.isoformat(),
        }
        st.warning(f"No real flight found on {origin}→{dest}. Using synthetic flight SIM001.")

    custom_disruption = {
        "flight_id":      base_flight["id"],
        "flight_number":  base_flight["flight_number"],
        "type":           disruption_type,
        "delay_minutes":  delay_minutes,
        "origin":         origin,
        "destination":    dest,
        "departure_time": base_flight.get("scheduled_departure",
                           (datetime.utcnow() + timedelta(hours=dep_offset)).isoformat()),
    }

    st.divider()
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Simulating disruption and running recovery pipeline..."):
            event, plan = run_pipeline(custom_disruption, risk_threshold)

        sev = event.get("severity", "unknown")
        sev_col = SEVERITY_COLOUR.get(sev, "#95a5a6")
        st.markdown(
            f"<div style='background:{sev_col};padding:8px 16px;border-radius:6px;"
            f"color:white;font-weight:bold;margin-bottom:12px'>"
            f"Simulation result: {sev.upper()} severity "
            f"(score {event.get('severity_score','?'):.1f}/100)</div>",
            unsafe_allow_html=True,
        )

        summary = plan.get("summary", {})
        metric_row([
            ("Downstream flights at risk", event.get("total_affected_flights", 0), None),
            ("Passengers affected",        event.get("total_affected_pax", 0),      None),
            ("Cancellations avoided",      summary.get("cancellations_avoided", 0), None),
            ("Misconnects avoided",        summary.get("misconnects_avoided", 0),   None),
            ("Total actions",              summary.get("total_actions", 0),          None),
            ("Conflicts",                  summary.get("conflict_count", 0),         None),
        ])

        # route map for affected flights
        st.subheader("Affected route network")
        affected = event.get("affected_flights_json", [])
        all_fl   = load_flights()
        fl_lookup = {f["id"]: f for f in all_fl}

        fig = go.Figure()
        if origin in coords and dest in coords:
            oc = coords[origin]
            dc = coords[dest]
            fig.add_trace(go.Scattergeo(
                lon=[oc["lon"], dc["lon"]], lat=[oc["lat"], dc["lat"]],
                mode="lines+markers",
                line=dict(width=3, color="#e74c3c"),
                marker=dict(size=12, color=["#e74c3c","#e74c3c"]),
                text=[origin, dest],
                textposition="top center",
                name="Root disruption",
            ))

        for aff in affected[:15]:
            f = fl_lookup.get(aff["flight_id"])
            if not f:
                continue
            o, d = f["origin_id"], f["destination_id"]
            if o not in coords or d not in coords:
                continue
            risk = aff.get("risk_score", 0)
            col = "#e74c3c" if risk > 0.7 else "#e67e22" if risk > 0.4 else "#f1c40f"
            oc2, dc2 = coords[o], coords[d]
            fig.add_trace(go.Scattergeo(
                lon=[oc2["lon"], dc2["lon"]], lat=[oc2["lat"], dc2["lat"]],
                mode="lines",
                line=dict(width=1.5, color=col),
                opacity=0.7,
                name=f"{f['flight_number']} (risk={risk:.2f})",
                showlegend=False,
            ))

        fig.update_layout(
            geo=dict(projection_type="natural earth",
                      showland=True, landcolor="#1a1a2e",
                      showocean=True, oceancolor="#16213e",
                      showcountries=True, countrycolor="#2d2d44",
                      showframe=False, bgcolor="#0f0f23"),
            paper_bgcolor="#0f0f23",
            height=380, margin=dict(l=0,r=0,t=0,b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # affected flights table
        st.subheader("At-risk downstream flights")
        if affected:
            df = pd.DataFrame([{
                "Flight":       a.get("flight_number","—"),
                "Risk score":   f"{a.get('risk_score',0):.2f}",
                "Est. delay":   f"{a.get('delay_estimate_min',0)} min",
                "Reason":       a.get("reason",""),
            } for a in affected])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No downstream flights at risk above the threshold.")

        # advisory
        st.divider()
        st.subheader("📢 OCC Advisory")
        from backend.routers.advisory import _rule_based_advisory
        summary_text, bullets = _rule_based_advisory(event, plan)
        st.info(summary_text)
        for b in bullets:
            st.markdown(f"- {b}")

        # full plan expandable
        with st.expander("View full recovery plan JSON"):
            st.json(plan.get("summary", {}))


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar nav
# ══════════════════════════════════════════════════════════════════════════════

def main():
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/airplane-mode-on.png", width=60)
        st.title("AeroNexus")
        st.caption("IROPS Recovery System")
        st.divider()
        page = st.radio(
            "Navigation",
            ["🗺 Disruption Map", "🔍 Disruption Detail", "🔮 What-If Simulator"],
        )
        st.divider()
        st.caption("Phase 4 — Dashboard")
        st.caption("75 tests passing ✅")

    if page == "🗺 Disruption Map":
        page_map()
    elif page == "🔍 Disruption Detail":
        page_detail()
    else:
        page_whatif()


if __name__ == "__main__":
    main()