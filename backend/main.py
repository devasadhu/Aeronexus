"""
backend/main.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routers import flights, disruptions, advisory, recovery, pdf_export, weather, occ_assistant


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from .db import init_db
        init_db()
    except Exception as e:
        print(f"DB init skipped (no DB configured): {e}")
    yield


app = FastAPI(
    title="AeroNexus IROPS API",
    description=(
        "Agentic flight disruption recovery system. "
        "Full pipeline: cascade prediction → multi-agent recovery → "
        "FAR 117 legality → cost estimation → OCC advisory."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(flights.router,        prefix="/flights",       tags=["Flights"])
app.include_router(disruptions.router,    prefix="/disruptions",   tags=["Disruptions"])
app.include_router(advisory.router,       prefix="/advisory",      tags=["Advisory"])
app.include_router(recovery.router,       prefix="/recovery",      tags=["Recovery & Audit"])
app.include_router(pdf_export.router,     prefix="/export",        tags=["Export"])
app.include_router(weather.router,        prefix="/weather",       tags=["Weather"])
app.include_router(occ_assistant.router,  prefix="/occ",           tags=["OCC Assistant"])


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "0.3.0"}


@app.get("/", tags=["System"])
def root():
    return {
        "system":  "AeroNexus IROPS Recovery Platform",
        "version": "0.3.0",
        "docs":    "/docs",
        "endpoints": {
            "flights":        "/flights",
            "disruptions":    "/disruptions",
            "recovery":       "/recovery/run?flight_id=<id>",
            "audit":          "/recovery/audit",
            "network_health": "/recovery/health",
            "advisory":       "/advisory/full-pipeline",
            "pdf_export":     "/export/pdf",
            "weather":        "/weather/synthetic",
            "occ_chat":       "/occ/chat",
        },
    }