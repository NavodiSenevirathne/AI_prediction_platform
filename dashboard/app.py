"""Monitoring dashboard for the real-time prediction platform.

Run locally:
    API_URL=http://localhost:8000 streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FEATURES  # noqa: E402
from src.data.generate import generate_customers  # noqa: E402


def _api_url() -> str:
    url = os.getenv("API_URL", "http://localhost:8000").strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        local = url.startswith(("localhost", "127.0.0.1", "api:", "api/")) or url == "api"
        url = ("http://" if local else "https://") + url
    return url


API_URL = _api_url()
TIMEOUT = 60  # free-tier hosts can take ~1 min to wake up

st.set_page_config(page_title="MLOps Monitoring", page_icon="📈", layout="wide")


@st.cache_data(ttl=5, show_spinner=False)
def get(path: str, **params) -> dict | list | None:
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.session_state["api_error"] = str(exc)
        return None


def post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return None


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Prediction platform")
    st.caption(f"API: `{API_URL}`")
    window = st.selectbox(
        "Monitoring window",
        [15, 60, 360, 1440, 10080],
        index=1,
        format_func=lambda m: {
            15: "Last 15 min",
            60: "Last hour",
            360: "Last 6 hours",
            1440: "Last 24 hours",
            10080: "Last 7 days",
        }[m],
    )
    drift_window = st.slider("Drift check: last N predictions", 100, 5000, 500, step=100)
    if st.button("Refresh now", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("Simulate live traffic")
    sim_n = st.slider("Requests", 50, 1000, 200, step=50)
    sim_drift = st.slider(
        "Data drift level", 0.0, 1.0, 0.0, step=0.1, help="0 = same as training data"
    )
    if st.button("Send traffic", type="primary", width="stretch"):
        customers = generate_customers(sim_n, drift=sim_drift)[FEATURES].to_dict("records")
        progress = st.progress(0.0, text="Sending requests...")
        for i in range(0, len(customers), 50):
            post("/predict/batch", {"customers": customers[i : i + 50]})
            progress.progress(min((i + 50) / len(customers), 1.0))
        progress.empty()
        st.success(f"Sent {sim_n} predictions (drift={sim_drift})")
        st.cache_data.clear()

health = get("/health")
if health is None:
    st.title("📈 MLOps Monitoring Dashboard")
    st.error(
        f"Cannot reach the API at {API_URL}. If it is hosted on a free tier it may be "
        f"waking up - wait a minute and press Refresh.\n\n"
        f"Details: {st.session_state.get('api_error', '')}"
    )
    st.stop()

summary = get("/monitoring/summary", window_minutes=window) or {}
info = get("/model/info") or {}

st.title("📈 MLOps Monitoring Dashboard")
st.caption(
    f"Model **v{info.get('version')}** · trained {info.get('trained_at', '?')} · "
    f"API status **{health['status']}**"
)

tab_overview, tab_drift, tab_model, tab_predict = st.tabs(
    ["Live traffic", "Data drift", "Model & retraining", "Try a prediction"]
)

# ------------------------------------------------------------------ overview
with tab_overview:
    c = st.columns(5)
    c[0].metric("Total predictions", f"{summary.get('total_predictions', 0):,}")
    c[1].metric("In window", f"{summary.get('window_predictions', 0):,}")
    c[2].metric(
        "Avg latency",
        f"{summary['avg_latency_ms']:.2f} ms" if summary.get("avg_latency_ms") is not None else "–",
    )
    c[3].metric(
        "P95 latency",
        f"{summary['p95_latency_ms']:.2f} ms" if summary.get("p95_latency_ms") is not None else "–",
    )
    c[4].metric(
        "Predicted churn rate",
        f"{summary['positive_rate']:.1%}" if summary.get("positive_rate") is not None else "–",
    )

    per_minute = pd.DataFrame(summary.get("per_minute", []))
    if per_minute.empty:
        st.info("No predictions in this window yet. Use **Send traffic** in the sidebar.")
    else:
        per_minute["minute"] = pd.to_datetime(per_minute["minute"])
        per_minute = per_minute.set_index("minute")
        left, right = st.columns(2)
        with left:
            st.subheader("Requests per minute")
            st.bar_chart(per_minute["requests"], height=260)
        with right:
            st.subheader("Predicted churn rate per minute")
            st.line_chart(per_minute["positive_rate"], height=260)
        left, right = st.columns(2)
        with left:
            st.subheader("Avg latency per minute (ms)")
            st.line_chart(per_minute["avg_latency_ms"], height=260)
        with right:
            st.subheader("Churn probability distribution")
            hist = pd.DataFrame(summary.get("probability_histogram", []))
            if not hist.empty:
                st.bar_chart(hist.set_index("bin")["count"], height=260)

    recent = get("/monitoring/recent", limit=20) or []
    if recent:
        st.subheader("Latest predictions")
        rows = [
            {
                "time": r["created_at"][11:19],
                "probability": round(r["probability"], 3),
                "churn": bool(r["prediction"]),
                "model": f"v{r['model_version']}",
                **{k: r["features"][k] for k in ("contract_type", "tenure_months", "monthly_charges")},
            }
            for r in recent
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ------------------------------------------------------------------ drift
with tab_drift:
    drift = get("/monitoring/drift", window=drift_window)
    if not drift or drift.get("status") != "ok":
        st.info(
            (drift or {}).get("detail", "Drift report unavailable.")
            + " Send some traffic from the sidebar first."
        )
    else:
        if drift["dataset_drift"]:
            st.error(
                f"⚠️ Dataset drift detected: {drift['n_drifted_features']} of "
                f"{drift['n_features']} features drifted (threshold "
                f"{drift['dataset_drift_share_threshold']:.0%}). "
                "The scheduled retraining pipeline will retrain and evaluate a new model."
            )
        else:
            st.success(
                f"✅ No dataset drift: {drift['n_drifted_features']} of "
                f"{drift['n_features']} features drifted."
            )
        c = st.columns(3)
        c[0].metric("Share of drifted features", f"{drift['share_drifted_features']:.0%}")
        c[1].metric("Reference rows", f"{drift['reference_rows']:,}")
        c[2].metric("Production rows compared", f"{drift['current_rows']:,}")

        feats = pd.DataFrame(
            [{"feature": k, **v} for k, v in drift["features"].items()]
        ).sort_values("psi", ascending=False)
        st.subheader("Population Stability Index per feature")
        st.caption(
            f"PSI ≥ {drift['psi_threshold']} = significant drift · 0.1–0.2 moderate · < 0.1 stable"
        )
        st.bar_chart(feats.set_index("feature")["psi"], height=300)
        cols = ["feature", "type", "psi", "test", "p_value", "drifted"]
        if "reference_mean" in feats:
            cols += ["reference_mean", "current_mean"]
        st.dataframe(feats[cols], width="stretch", hide_index=True)

# ------------------------------------------------------------------ model
with tab_model:
    metrics = info.get("metrics", {})
    c = st.columns(5)
    for col, key in zip(c, ["roc_auc", "f1", "precision", "recall", "accuracy"], strict=False):
        col.metric(key.upper().replace("_", " "), metrics.get(key, "–"))

    history = pd.DataFrame(info.get("history", []))
    if not history.empty:
        st.subheader("Model version history (automated retraining)")
        if len(history) > 1:
            st.line_chart(history.set_index("version")[["roc_auc", "f1"]], height=260)
        st.dataframe(history.iloc[::-1], width="stretch", hide_index=True)

    with st.expander("Full model metadata"):
        st.json(info)

# ------------------------------------------------------------------ predict
with tab_predict:
    st.subheader("Score a single customer in real time")
    with st.form("predict"):
        a, b, c = st.columns(3)
        payload = {
            "age": a.number_input("Age", 18, 100, 34),
            "tenure_months": a.number_input("Tenure (months)", 0, 120, 5),
            "monthly_charges": a.number_input("Monthly charges", 0.0, 500.0, 89.5),
            "total_charges": b.number_input("Total charges", 0.0, 50_000.0, 447.5),
            "num_support_calls": b.number_input("Support calls", 0, 50, 4),
            "avg_daily_usage_gb": b.number_input("Avg daily usage (GB)", 0.0, 200.0, 6.2),
            "contract_type": c.selectbox("Contract", ["month_to_month", "one_year", "two_year"]),
            "internet_service": c.selectbox("Internet", ["fiber", "dsl", "none"]),
            "payment_method": c.selectbox(
                "Payment", ["electronic_check", "credit_card", "bank_transfer"]
            ),
        }
        if st.form_submit_button("Predict", type="primary"):
            result = post("/predict", payload)
            if result:
                st.cache_data.clear()
                colour = {"high": "🔴", "medium": "🟠", "low": "🟢"}[result["risk_level"]]
                st.metric(
                    f"{colour} Churn probability ({result['risk_level']} risk)",
                    f"{result['churn_probability']:.1%}",
                )
                st.json(result)
