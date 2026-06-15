# AeroNexus — Agentic IROPS Recovery System

End-to-end multi-agent AI platform for airline irregular operations (IROPS) recovery.

## What it does

- Detects & predicts flight disruptions and downstream cascade propagation
- Plans recovery across fleet, crew, and passengers via specialised agents
- Explains decisions through natural-language OCC-style advisories and an ops dashboard

## Architecture

```
Ingestion Layer        →  Disruption Intelligence  →  Multi-Agent Core  →  Advisory + Dashboard
(BTS, ADS-B, METAR)       (XGBoost cascade model)     (Python agents)      (Groq LLM + Streamlit)
```

## Project structure

```
aeronexus/
├── backend/          FastAPI app, SQLAlchemy models, Pydantic schemas
├── agents/           FleetAgent, CrewAgent, PassengerAgent, CoordinatorAgent
├── ml/               Feature builder, cascade model, severity scorer
├── ingestion/        BTS loader, airport/route loader, weather loader
├── synthetic/        Crew, aircraft, passenger data generators
├── dashboard/        Streamlit ops dashboard (orthographic globe, 3 pages)
├── tests/            pytest suite (75 tests passing)
└── data/             raw/ and processed/ data files
```

## Quick start

```bash
# 1. Create and activate venv
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Load airport/route network
python -m ingestion.load_airports --hub-only --no-db

# 4. Generate synthetic operational data
python -m synthetic.gen_crew --no-db
python -m synthetic.gen_aircraft --no-db
python -m ingestion.load_bts --synthetic --sample 5000 --no-db
python -m synthetic.gen_passengers --flights-json data/processed/flights.json --no-db

# 5. Build features and train cascade model
python ml/feature_builder.py
python -m ml.cascade_model --train

# 6. Run demo
python -m ml.cascade_model --demo
python ml/severity_scorer.py

# 7. Run tests
pytest tests/ -v

# 8. Launch dashboard
streamlit run dashboard/app.py
```

## Groq LLM advisory (optional)

The advisory router calls Groq's LLaMA-3.3-70b for natural-language OCC advisories.
Without a key it silently falls back to rule-based text.

```bash
# Create a .env file in the project root:
GROQ_API_KEY=gsk_...
```

Then install python-dotenv and load it at startup, or export the variable directly:

```bash
export GROQ_API_KEY=gsk_...   # macOS/Linux
set GROQ_API_KEY=gsk_...      # Windows CMD
```

## With PostgreSQL

```bash
export DATABASE_URL=postgresql://user:pass@localhost:5432/aeronexus
# Drop --no-db flags from all ingestion commands above
```

## Tech stack

| Layer     | Tech                                      |
|-----------|-------------------------------------------|
| API       | FastAPI + SQLAlchemy + PostgreSQL         |
| Agents    | Python multi-agent (Fleet/Crew/Pax/Coord) |
| ML        | XGBoost, scikit-learn, NetworkX           |
| LLM       | Groq (LLaMA-3.3-70b, optional)            |
| Dashboard | Streamlit + Plotly (orthographic globe)   |
| Tests     | pytest — 75 tests passing                 |

## ML model notes

The cascade model (XGBoost) predicts which downstream flights are at risk when a root disruption occurs.
Expected validation AUC on synthetic data: **0.70–0.85**, driven by network centrality, hub flags,
downstream airport pressure, and historical delay rates on the route.

Note: an earlier version of the training data had label leakage (`risk_score_gt` and zero-delay
negative examples). This was fixed in `feature_builder.py` — negative examples now receive real
disruption context sampled from the disruption pool, and `risk_score_gt` was removed from features.

## Status

- ✅ Phase 1: DB schema, airport/route graph, synthetic data
- ✅ Phase 2: Disruption intelligence engine (cascade model + severity scorer)
- ✅ Phase 3: Multi-agent recovery core (Fleet, Crew, Passenger, Coordinator)
- ✅ Phase 4: Advisory (Groq LLM + rule-based fallback) + Streamlit dashboard