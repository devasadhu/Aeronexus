"""
ml/explainability.py

SHAP-based explainability for the cascade disruption model.

Functions:
  explain_prediction()   — SHAP values for a single flight prediction
  explain_batch()        — SHAP summary for a set of predictions
  shap_summary_df()      — DataFrame of mean |SHAP| per feature (for bar chart)
  waterfall_data()       — Single-prediction waterfall data (for dashboard)
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap

from ml.cascade_model import load_model, FEATURE_COLS

log = logging.getLogger(__name__)

SHAP_CACHE_PATH = Path("data/processed/shap_explainer.pkl")


# ── Explainer (cached) ────────────────────────────────────────────────────────

def get_explainer(model=None):
    """Build or load a TreeExplainer for the cascade model."""
    import pickle
    if SHAP_CACHE_PATH.exists():
        with open(SHAP_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    if model is None:
        model = load_model()

    explainer = shap.TreeExplainer(model)

    SHAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SHAP_CACHE_PATH, "wb") as f:
        pickle.dump(explainer, f)

    log.info("SHAP TreeExplainer cached.")
    return explainer


# ── Single prediction explanation ─────────────────────────────────────────────

def explain_prediction(feature_dict: dict) -> dict:
    """
    Given a flat feature dict (output of extract_features()),
    return SHAP values and a human-readable explanation.

    Returns:
    {
        "risk_score": float,
        "base_value": float,
        "shap_values": {feature: shap_val, ...},
        "top_drivers": [(feature, shap_val, direction), ...],
        "explanation": str,
    }
    """
    model     = load_model()
    explainer = get_explainer(model)

    df = pd.DataFrame([feature_dict])
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    X = df[FEATURE_COLS].fillna(0).astype(float)

    shap_values = explainer.shap_values(X)
    # For XGBClassifier TreeExplainer returns array of shape (1, n_features)
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

    risk_score = float(model.predict_proba(X)[0, 1])
    base_value = float(explainer.expected_value
                       if not isinstance(explainer.expected_value, (list, np.ndarray))
                       else explainer.expected_value[1])

    shap_dict = {feat: round(float(val), 4)
                 for feat, val in zip(FEATURE_COLS, sv)}

    # top 5 drivers by absolute SHAP
    sorted_drivers = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_drivers = [
        (feat, val, "increases risk" if val > 0 else "reduces risk")
        for feat, val in sorted_drivers
    ]

    explanation = _build_explanation(feature_dict, top_drivers, risk_score)

    return {
        "risk_score":  round(risk_score, 3),
        "base_value":  round(base_value, 4),
        "shap_values": shap_dict,
        "top_drivers": top_drivers,
        "explanation": explanation,
    }


# ── Batch SHAP summary ────────────────────────────────────────────────────────

def shap_summary_df(feature_dicts: list) -> pd.DataFrame:
    """
    Compute mean absolute SHAP value per feature across a batch.
    Returns DataFrame sorted by importance (for bar chart).
    """
    if not feature_dicts:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])

    model     = load_model()
    explainer = get_explainer(model)

    df = pd.DataFrame(feature_dicts)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    X = df[FEATURE_COLS].fillna(0).astype(float)

    shap_values = explainer.shap_values(X)
    sv = shap_values if not isinstance(shap_values, list) else shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    result = pd.DataFrame({
        "feature":        FEATURE_COLS,
        "mean_abs_shap":  np.round(mean_abs, 4),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return result


def waterfall_data(feature_dict: dict) -> dict:
    """
    Returns data needed to render a waterfall chart for one prediction.
    Sorted by absolute SHAP, top 8 features shown, rest aggregated.
    """
    result  = explain_prediction(feature_dict)
    sv      = result["shap_values"]
    sorted_ = sorted(sv.items(), key=lambda x: abs(x[1]), reverse=True)

    top_n   = 8
    top     = sorted_[:top_n]
    rest    = sorted_[top_n:]
    rest_sum = sum(v for _, v in rest)

    features = [f for f, _ in top] + (["other features"] if rest else [])
    values   = [v for _, v in top] + ([round(rest_sum, 4)] if rest else [])

    return {
        "features":   features,
        "values":     values,
        "base_value": result["base_value"],
        "risk_score": result["risk_score"],
    }


# ── Human-readable explanation ────────────────────────────────────────────────

_FEATURE_LABELS = {
    "delay_minutes":      "root delay magnitude",
    "delay_category":     "delay severity category",
    "hour":               "departure hour",
    "day_of_week":        "day of week",
    "both_hubs":          "hub-to-hub route",
    "is_hub_origin":      "hub origin airport",
    "is_hub_dest":        "hub destination airport",
    "bc_origin":          "origin betweenness centrality",
    "bc_dest":            "destination betweenness centrality",
    "degree_origin":      "origin airport connectivity",
    "degree_dest":        "destination airport connectivity",
    "downstream_count":   "downstream departure pressure",
    "downstream_next_hr": "next-hour downstream pressure",
    "hist_delay_rate":    "historical delay rate on route",
    "hist_mean_delay":    "historical mean delay on route",
    "is_weather":         "weather disruption flag",
    "disruption_type":    "disruption type",
    "alt_routes":         "number of alternative routes",
    "is_long_haul":       "long-haul flight",
    "distance_km":        "route distance",
    "is_peak_hour":       "peak hour flag",
    "is_weekend":         "weekend flag",
}


def _build_explanation(features: dict, top_drivers: list, risk_score: float) -> str:
    level = ("HIGH" if risk_score > 0.7 else
             "MODERATE" if risk_score > 0.4 else "LOW")

    parts = [f"Cascade risk is {level} ({risk_score:.0%})."]
    for feat, val, direction in top_drivers[:3]:
        label = _FEATURE_LABELS.get(feat, feat.replace("_", " "))
        parts.append(f"{label.capitalize()} {direction} (SHAP={val:+.3f}).")

    return " ".join(parts)


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle as pkl
    from ml.feature_builder import (
        load_graph, build_historical_rates, build_downstream_index,
        extract_features, compute_centrality,
    )
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    flights     = _json.loads(Path("data/processed/flights.json").read_text())
    disruptions = _json.loads(Path("data/processed/disruptions_seed.json").read_text())
    G           = load_graph()

    hist_rates     = build_historical_rates(flights)
    downstream_idx = build_downstream_index(flights)
    centrality     = compute_centrality(G) if G else {}

    f = flights[0]
    d = disruptions[0]
    feats = extract_features(f, d, G, centrality, hist_rates, downstream_idx)

    result = explain_prediction(feats)
    log.info(f"Risk score: {result['risk_score']}")
    log.info(f"Explanation: {result['explanation']}")
    log.info("Top drivers:")
    for feat, val, direction in result["top_drivers"]:
        log.info(f"  {feat:<25s}  {val:+.4f}  ({direction})")

    summary = shap_summary_df([feats])
    log.info("\nGlobal feature importance (mean |SHAP|):")
    print(summary.head(10).to_string(index=False))