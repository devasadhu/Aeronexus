"""
ml/cascade_model.py

Trains and serves the cascade disruption prediction model.

  train()   — builds features from flights/disruptions, trains XGBoost,
              saves model + feature list to disk.
  predict_cascade(disruption_event, flights, G) — returns list of
              AffectedFlight dicts with risk_score and delay_estimate.

Usage:
    python -m ml.cascade_model --train
    python -m ml.cascade_model --demo
"""

import sys, json, pickle, logging, argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, average_precision_score
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH    = Path("data/processed/cascade_model.xgb")
FEATURES_PATH = Path("data/processed/feature_columns.json")
TRAIN_PATH    = Path("data/processed/training_features.parquet")

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_peak_hour", "is_weekend", "is_night",
    "is_hub_origin", "is_hub_dest", "both_hubs",
    "distance_km", "duration_min", "is_long_haul",
    "bc_origin", "bc_dest", "degree_origin", "degree_dest", "alt_routes",
    "delay_minutes", "delay_category", "is_cancelled", "is_weather",
    "disruption_type",
    "hist_delay_rate", "hist_mean_delay", "hist_sample_size",
    "downstream_count", "downstream_next_hr",
]


# ── Training ──────────────────────────────────────────────────────────────────

def train(df: Optional[pd.DataFrame] = None) -> xgb.XGBClassifier:
    if df is None:
        if not TRAIN_PATH.exists():
            log.info("Training features not found. Building now...")
            from ml.feature_builder import (
                load_graph, build_training_dataset
            )
            flights     = json.loads(Path("data/processed/flights.json").read_text())
            disruptions = json.loads(Path("data/processed/disruptions_seed.json").read_text())
            G           = load_graph()
            df = build_training_dataset(flights, disruptions, G)
        else:
            df = pd.read_parquet(TRAIN_PATH)
            log.info(f"Loaded training features: {len(df)} rows")

    # ensure all feature cols exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLS].fillna(0).astype(float)
    y = df["impacted"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # evaluation
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.4).astype(int)

    auc  = roc_auc_score(y_val, y_prob)
    ap   = average_precision_score(y_val, y_prob)
    log.info(f"Validation AUC={auc:.4f}  AP={ap:.4f}")
    log.info("\n" + classification_report(y_val, y_pred, target_names=["not_impacted","impacted"]))

    # feature importance top-10
    importance = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:10]
    log.info("Top-10 features:")
    for feat, imp in importance:
        log.info(f"  {feat:<30s}  {imp:.4f}")

    # save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    Path(FEATURES_PATH).write_text(json.dumps(FEATURE_COLS))
    log.info(f"Model saved to {MODEL_PATH}")

    return model


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model() -> xgb.XGBClassifier:
    if not MODEL_PATH.exists():
        log.info("No saved model found. Training now...")
        return train()
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def predict_cascade(
    disruption: dict,
    candidate_flights: list,
    G,
    hist_rates: dict,
    downstream_idx: dict,
    risk_threshold: float = 0.30,
    max_results: int = 20,
) -> list:
    """
    Given a disruption dict and a list of candidate downstream flights,
    return sorted list of AffectedFlight dicts.

    disruption = {
        "flight_id": ..., "type": "weather"|..., "delay_minutes": int,
        "origin": IATA, "destination": IATA, "departure_time": ISO
    }
    """
    from ml.feature_builder import (
        extract_features, compute_centrality, DISRUPTION_TYPE_MAP
    )

    if not candidate_flights:
        return []

    model      = load_model()
    centrality = compute_centrality(G) if G else {}

    rows = []
    for f in candidate_flights:
        feats = extract_features(f, disruption, G, centrality,
                                  hist_rates, downstream_idx)
        rows.append(feats)

    df = pd.DataFrame(rows)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURE_COLS].fillna(0).astype(float)
    probs = model.predict_proba(X)[:, 1]

    results = []
    for f, prob in zip(candidate_flights, probs):
        if prob < risk_threshold:
            continue

        # estimate downstream delay from risk score + root delay
        base_delay  = disruption.get("delay_minutes", 0)
        delay_est   = int(base_delay * prob + 15 * prob)

        results.append({
            "flight_id":           f["id"],
            "flight_number":       f.get("flight_number", ""),
            "risk_score":          round(float(prob), 3),
            "delay_estimate_min":  delay_est,
            "reason":              _risk_reason(disruption, prob),
        })

    results.sort(key=lambda x: x["risk_score"], reverse=True)
    return results[:max_results]


def _risk_reason(disruption: dict, prob: float) -> str:
    dtype = disruption.get("type", "unknown")
    delay = disruption.get("delay_minutes", 0)
    if prob > 0.8:
        return f"High cascade risk: root {dtype} disruption, {delay}min delay"
    elif prob > 0.5:
        return f"Moderate cascade risk: downstream of {dtype} event"
    else:
        return f"Low cascade risk: marginal exposure to {dtype} event"


# ── Demo ──────────────────────────────────────────────────────────────────────

def demo():
    import pickle as pkl
    G = None
    graph_path = Path("data/processed/flight_graph.gpickle")
    if graph_path.exists():
        with open(graph_path, "rb") as f:
            G = pkl.load(f)

    flights     = json.loads(Path("data/processed/flights.json").read_text())
    disruptions = json.loads(Path("data/processed/disruptions_seed.json").read_text())

    from ml.feature_builder import build_historical_rates, build_downstream_index
    hist_rates     = build_historical_rates(flights)
    downstream_idx = build_downstream_index(flights)

    # pick first disruption as demo
    disruption = disruptions[0]
    log.info(f"\nDemo disruption: flight {disruption['flight_number']} "
             f"| type={disruption['type']} | delay={disruption['delay_minutes']}min "
             f"| {disruption['origin']} → {disruption['destination']}")

    # candidates: flights departing from disruption's destination
    candidates = [f for f in flights
                   if f["origin_id"] == disruption["destination"]
                   and f["status"] not in ("cancelled",)][:30]

    results = predict_cascade(disruption, candidates, G, hist_rates,
                               downstream_idx, risk_threshold=0.2)

    log.info(f"\nCascade results ({len(results)} at-risk flights):")
    for r in results[:5]:
        log.info(f"  {r['flight_number']:10s}  risk={r['risk_score']:.2f}  "
                 f"delay_est={r['delay_estimate_min']}min  {r['reason']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--demo",  action="store_true")
    args = parser.parse_args()

    if args.train:
        train()
    if args.demo:
        demo()
    if not args.train and not args.demo:
        parser.print_help()


if __name__ == "__main__":
    main()