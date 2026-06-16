"""
backend/routers/occ_assistant.py

OCC Assistant — Groq-powered chatbot grounded in live AeroNexus system state.

Unlike a generic LLM chat, this assistant:
  1. Pulls live context before every response:
     - Current network health score
     - Active disruptions + severity
     - Latest recovery plan summaries (from audit trail)
     - Synthetic weather snapshot
  2. Injects that context into the system prompt so answers are
     grounded in YOUR data, not hallucinated airline knowledge.
  3. Supports structured tool-style queries:
     "Which crew are near their duty limit?"
     "What's the cheapest recovery for ORD?"
     "How many passengers are stranded right now?"
     "Show me cascade risk for JFK"

Endpoint:
  POST /chat
  Body: { "message": "...", "history": [...] }
  Returns: { "reply": "...", "context_used": {...}, "sources": [...] }
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from typing import Optional
import json, os
from pathlib import Path

router = APIRouter()

OCC_SYSTEM_PROMPT = """You are the AeroNexus OCC Assistant — an AI embedded in an airline 
Operations Control Centre (OCC) decision support system. You have access to live system data 
injected below. Use it to answer operator questions accurately and concisely.

Persona:
- Speak like an experienced OCC duty manager: direct, precise, no fluff
- Use aviation terminology (FDP, MEL, ATC, IROPS, LVP, PNR, OAG, etc.)
- Lead with the number/fact, then the explanation
- Flag risks clearly: use ALERT, WARNING, or NOTE prefixes when appropriate
- Never say "I don't have access to" — you have the live context below

Capabilities you can speak to:
- Flight network status and cascade risk
- Recovery plan decisions (fleet, crew, pax)
- FAR 117 crew legality and duty limits
- Weather impact and airport capacity
- Cost of disruption vs recovery
- Audit trail: what changed between plan versions
- Passenger reaccommodation status

Always ground answers in the LIVE CONTEXT section. If data is missing, say so and suggest 
what the operator should check."""


def _build_live_context() -> dict:
    """Assemble live system state for injection into chatbot context."""
    context = {}

    # flights
    fp = Path("data/processed/flights.json")
    flights = json.loads(fp.read_text()) if fp.exists() else []
    context["total_flights"]  = len(flights)
    context["delayed_flights"]  = sum(1 for f in flights if f.get("delay_minutes", 0) > 15)
    context["cancelled_flights"] = sum(1 for f in flights if f.get("status") == "cancelled")
    context["avg_delay_min"]  = round(
        sum(f.get("delay_minutes", 0) for f in flights) / max(len(flights), 1), 1
    )

    # disruptions
    dp = Path("data/processed/disruptions_seed.json")
    disruptions = json.loads(dp.read_text()) if dp.exists() else []
    context["active_disruptions"] = len(disruptions)
    if disruptions:
        context["disruption_sample"] = [
            {
                "flight_id":  d.get("flight_id"),
                "type":       d.get("disruption_type"),
                "delay_min":  d.get("delay_minutes", 0),
                "origin":     d.get("origin"),
                "dest":       d.get("destination"),
            }
            for d in disruptions[:5]
        ]

    # crew
    cp = Path("data/processed/crew_roster.json")
    crew = json.loads(cp.read_text()) if cp.exists() else []
    context["total_crew"] = len(crew)
    near_limit = [
        c for c in crew
        if c.get("current_duty_hours", 0) >= c.get("max_duty_hours", 14) * 0.85
    ]
    context["crew_near_duty_limit"] = len(near_limit)
    if near_limit:
        context["crew_near_limit_sample"] = [
            {
                "name":        c["name"],
                "role":        c.get("role"),
                "duty_hours":  c.get("current_duty_hours"),
                "max_hours":   c.get("max_duty_hours", 14),
                "location":    c.get("current_location_id"),
            }
            for c in near_limit[:5]
        ]

    # passengers
    pp = Path("data/processed/passengers.json")
    passengers = json.loads(pp.read_text()) if pp.exists() else []
    context["total_passengers"] = len(passengers)

    # audit trail — latest plans
    ap = Path("data/processed/audit_trail.json")
    if ap.exists():
        trail = json.loads(ap.read_text())
        latest = sorted(trail, key=lambda r: r["timestamp"], reverse=True)[:5]
        context["recent_plans"] = [
            {
                "flight_id":  r["flight_id"],
                "version":    r["version"],
                "timestamp":  r["timestamp"],
                "trigger":    r["trigger"],
                "summary":    r["plan_summary"],
                "diff":       r.get("diff_from_prev", {}),
            }
            for r in latest
        ]

    # network health
    try:
        from backend.audit_trail import get_network_health
        context["network_health"] = get_network_health(flights)
    except Exception:
        pass

    # synthetic weather
    try:
        from backend.routers.weather import synthetic_weather
        snap = synthetic_weather(seed=datetime.now(timezone.utc).hour)
        context["weather_summary"] = snap["network_summary"]
        context["lvp_airports"]    = snap["network_summary"].get("lvp_airports", [])
    except Exception:
        pass

    return context


def _format_context_for_prompt(ctx: dict) -> str:
    """Convert live context dict to a readable prompt section."""
    lines = ["=== LIVE SYSTEM CONTEXT ==="]

    h = ctx.get("network_health", {})
    lines.append(f"Network Health Score: {h.get('score', '?')}/100")
    lines.append(f"Flights: {ctx.get('total_flights',0)} total | "
                 f"{ctx.get('delayed_flights',0)} delayed | "
                 f"{ctx.get('cancelled_flights',0)} cancelled | "
                 f"avg delay {ctx.get('avg_delay_min',0)}min")
    lines.append(f"Active Disruptions: {ctx.get('active_disruptions',0)}")

    if ctx.get("disruption_sample"):
        lines.append("Top Disruptions:")
        for d in ctx["disruption_sample"]:
            lines.append(f"  • {d['flight_id']} | {d['type']} | "
                         f"{d['origin']}→{d['dest']} | +{d['delay_min']}min")

    lines.append(f"Crew: {ctx.get('total_crew',0)} total | "
                 f"{ctx.get('crew_near_duty_limit',0)} near duty limit")

    if ctx.get("crew_near_limit_sample"):
        lines.append("Crew Near Limit:")
        for c in ctx["crew_near_limit_sample"]:
            pct = round(c['duty_hours'] / c['max_hours'] * 100) if c['max_hours'] else 0
            lines.append(f"  • {c['name']} ({c['role']}) @ {c['location']} — "
                         f"{c['duty_hours']}h/{c['max_hours']}h ({pct}%)")

    lines.append(f"Passengers: {ctx.get('total_passengers',0)} in system")

    ws = ctx.get("weather_summary", {})
    if ws:
        lines.append(f"Weather: avg capacity {ws.get('avg_capacity_pct','?')}% | "
                     f"LVP active at: {ws.get('lvp_airports', []) or 'none'}")

    if ctx.get("recent_plans"):
        lines.append("Recent Recovery Plans:")
        for p in ctx["recent_plans"]:
            diff_note = ""
            if p.get("diff"):
                changes = [f"{k}: {v['from']}→{v['to']}" for k, v in p["diff"].items()]
                diff_note = " | Δ " + ", ".join(changes[:3])
            lines.append(f"  • {p['flight_id']} v{p['version']} "
                         f"@ {p['timestamp'][:16]} [{p['trigger']}]{diff_note}")

    lines.append("=== END CONTEXT ===")
    return "\n".join(lines)


def _call_groq_chat(
    system_prompt: str,
    history: list[dict],
    user_message: str,
) -> Optional[str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        import groq
        client = groq.Groq(api_key=api_key)

        messages = [{"role": "system", "content": system_prompt}]
        for h in history[-6:]:   # last 6 turns for context window
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = messages,
            max_tokens  = 500,
            temperature = 0.2,   # low temp = factual, less creative
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Groq error: {e}]"


def _rule_based_reply(message: str, ctx: dict) -> str:
    """Fallback reply when Groq is not configured."""
    msg = message.lower()

    if any(w in msg for w in ["health", "status", "overview", "summary"]):
        h = ctx.get("network_health", {})
        return (
            f"Network health score: {h.get('score', '?')}/100. "
            f"{ctx.get('delayed_flights',0)} flights delayed, "
            f"{ctx.get('cancelled_flights',0)} cancelled, "
            f"avg delay {ctx.get('avg_delay_min',0)}min. "
            f"{ctx.get('active_disruptions',0)} active disruption(s)."
        )

    if any(w in msg for w in ["crew", "duty", "far", "limit"]):
        n = ctx.get("crew_near_duty_limit", 0)
        sample = ctx.get("crew_near_limit_sample", [])
        if n == 0:
            return "No crew members are currently near their FAR 117 duty limit."
        names = ", ".join(f"{c['name']} ({c['duty_hours']}h/{c['max_hours']}h)"
                          for c in sample[:3])
        return f"{n} crew member(s) at ≥85% duty limit: {names}."

    if any(w in msg for w in ["weather", "lvp", "capacity", "ifr", "taf"]):
        ws = ctx.get("weather_summary", {})
        lvp = ctx.get("lvp_airports", [])
        return (
            f"Network avg airport capacity: {ws.get('avg_capacity_pct','?')}%. "
            f"LVP active at: {', '.join(lvp) if lvp else 'none'}. "
            f"{ws.get('restricted_count',0)} airport(s) below 80% capacity."
        )

    if any(w in msg for w in ["passenger", "pax", "stranded", "rebook"]):
        return (
            f"{ctx.get('total_passengers',0)} passengers in system. "
            "Check the latest recovery plan for stranded pax count."
        )

    if any(w in msg for w in ["disruption", "cascade", "affected"]):
        n = ctx.get("active_disruptions", 0)
        sample = ctx.get("disruption_sample", [])
        if not sample:
            return f"{n} active disruption(s). No details available."
        top = sample[0]
        return (
            f"{n} active disruption(s). Most recent: "
            f"{top['flight_id']} ({top['type']}, {top['origin']}→{top['dest']}, "
            f"+{top['delay_min']}min)."
        )

    return (
        "I can answer questions about: network health, crew duty limits, "
        "weather/LVP status, active disruptions, passenger reaccommodation, "
        "and recovery plan history. What would you like to know?"
    )


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/chat")
def occ_chat(body: dict):
    """
    Body: {
      "message": "Which crew are near their duty limit?",
      "history": [{"role": "user", "content": "..."}, {"role": "assistant", ...}]
    }
    Returns: {
      "reply": "...",
      "context_used": { summary of what was injected },
      "llm_used": bool,
      "timestamp": ISO
    }
    """
    message = body.get("message", "").strip()
    history = body.get("history", [])

    if not message:
        raise HTTPException(400, detail="message field required")

    # build live context
    ctx            = _build_live_context()
    context_prompt = _format_context_for_prompt(ctx)
    full_system    = f"{OCC_SYSTEM_PROMPT}\n\n{context_prompt}"

    # try Groq first
    reply    = _call_groq_chat(full_system, history, message)
    llm_used = reply is not None and not reply.startswith("[Groq error")

    if not llm_used:
        reply = _rule_based_reply(message, ctx)

    # context summary for transparency
    ctx_summary = {
        "network_health_score":  ctx.get("network_health", {}).get("score"),
        "active_disruptions":    ctx.get("active_disruptions", 0),
        "crew_near_limit":       ctx.get("crew_near_duty_limit", 0),
        "lvp_airports":          ctx.get("lvp_airports", []),
        "recent_plan_count":     len(ctx.get("recent_plans", [])),
    }

    return {
        "reply":        reply,
        "context_used": ctx_summary,
        "llm_used":     llm_used,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }