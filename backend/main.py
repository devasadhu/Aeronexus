"""
backend/main.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routers import flights, disruptions, advisory


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: create tables if needed
    try:
        from .db import init_db
        init_db()
    except Exception as e:
        print(f"DB init skipped (no DB configured): {e}")
    yield


app = FastAPI(
    title="AeroNexus IROPS API",
    description="Agentic flight disruption recovery system",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(flights.router,     prefix="/flights",     tags=["flights"])
app.include_router(disruptions.router, prefix="/disruptions", tags=["disruptions"])
app.include_router(advisory.router,    prefix="/advisory",    tags=["advisory"])


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}