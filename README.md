# AeroNexus — Agentic IROPS Recovery System

End-to-end multi-agent AI platform for airline irregular operations (IROPS) recovery.

## What it does

- **Detects & predicts** flight disruptions and downstream cascade propagation
- **Plans recovery** across fleet, crew, and passengers via specialised agents
- **Explains decisions** through natural-language OCC-style advisories and an ops dashboard

## Architecture

```
Ingestion Layer        →  Disruption Intelligence  →  Multi-Agent Core  →  Advisory + Dashboard
(BTS, ADS-B, METAR)       (XGBoost cascade model)     (LangGraph agents)   (Groq LLM + Streamlit)
```

## Project structure

```
aeronexus/
├── backend/          FastAPI app, SQLAlchemy models, Pydantic schemas
├── agents/           FleetAgent, CrewAgent, PassengerAgent, CoordinatorAgent
├── ml/               Feature builder, cascade model, severity scorer
├── ingestion/        BTS loader, airport/route loader, weather loader
├── synthetic/        Crew, aircraft, passenger data generators
├── dashboard/        Streamlit ops dashboard
├── tests/            pytest suite (48 tests)
└── data/             raw/ and processed/ data files
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Load airport/route network
python -m ingestion.load_airports --hub-only --no-db

# 3. Generate synthetic operational data
python -m synthetic.gen_crew --no-db
python -m synthetic.gen_aircraft --no-db
python -m synthetic.gen_passengers --no-db

# 4. Generate synthetic flight data
python -m ingestion.load_bts --synthetic --sample 5000 --no-db

# 5. Build features and train cascade model
python ml/feature_builder.py
python -m ml.cascade_model --train

# 6. Run demo
python -m ml.cascade_model --demo
python ml/severity_scorer.py

# 7. Run tests
pytest tests/ -v
```

## With PostgreSQL

```bash
# Set connection string
export DATABASE_URL=postgresql://user:pass@localhost:5432/aeronexus

# Run all ingestion with DB seeding (drop --no-db flags)
python -m ingestion.load_airports --hub-only
python -m synthetic.gen_crew
python -m synthetic.gen_aircraft
python -m ingestion.load_bts --synthetic --sample 5000
```

## Tech stack

| Layer | Tech |
|---|---|
| API | FastAPI + SQLAlchemy + PostgreSQL |
| Agents | LangGraph |
| ML | XGBoost, scikit-learn, NetworkX |
| LLM | Groq (LLaMA 3) |
| Dashboard | Streamlit + Plotly |
| Tests | pytest (48 tests) |

## Status

- [x] Phase 1: DB schema, airport/route graph, synthetic data
- [x] Phase 2: Disruption intelligence engine (cascade model + severity scorer)
- [ ] Phase 3: Multi-agent recovery core
- [ ] Phase 4: NL advisory + dashboard
- [ ] Phase 5: Evaluation + polish