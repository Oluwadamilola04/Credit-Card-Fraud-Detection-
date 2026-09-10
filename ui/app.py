from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


def _default_api_url() -> str:
    try:
        configured_url = st.secrets.get("API_URL")
    except Exception:
        configured_url = None

    return str(
        configured_url or os.getenv("API_URL", "http://127.0.0.1:8000")
    ).rstrip("/")


DEFAULT_API_URL = _default_api_url()
SAMPLE_TRANSACTION = {
    "Time": 0.0,
    "V1": -1.0,
    "V2": -0.5,
    "V3": 0.2,
    "V4": 1.1,
    "V5": -0.2,
    "V6": 0.0,
    "V7": 0.8,
    "V8": -0.4,
    "V9": 0.3,
    "V10": -0.1,
    "V11": 0.5,
    "V12": -0.6,
    "V13": 0.7,
    "V14": -0.2,
    "V15": 0.3,
    "V16": -0.9,
    "V17": 0.1,
    "V18": -0.3,
    "V19": 0.2,
    "V20": 0.0,
    "V21": -0.1,
    "V22": 0.4,
    "V23": 0.2,
    "V24": -0.7,
    "V25": 0.5,
    "V26": 0.8,
    "V27": -0.2,
    "V28": 0.1,
    "Amount": 50.0,
}


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _get_json(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def _render_risk_result(probability: float, risk_label: str, decision: str) -> None:
    risk_color = {
        "Low risk": "#16a34a",
        "Moderate risk": "#d97706",
        "High risk": "#dc2626",
    }[risk_label]
    meter_width = max(2, min(100, probability * 100))

    st.markdown(
        f"""
        <div style="margin: 0.5rem 0 1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>Risk meter</strong>
                <strong style="color:{risk_color};">{probability:.1%}</strong>
            </div>
            <div style="background:#e5e7eb; border-radius:999px; height:14px; margin-top:0.4rem;">
                <div style="background:{risk_color}; border-radius:999px; height:14px; width:{meter_width}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if risk_label == "Low risk":
        st.success(f"{risk_label} | {decision}")
    elif risk_label == "Moderate risk":
        st.warning(f"{risk_label} | {decision}")
    else:
        st.error(f"{risk_label} | {decision}")


st.set_page_config(page_title="Fraud Predictor Demo", page_icon="🛡️", layout="wide")

st.title("Fraud Detection Demo")
st.caption("A simple fraud-risk scoring application built to showcase an end-to-end ML workflow.")

st.markdown(
    """
    ### What this app does
    This demo predicts whether a credit card transaction looks suspicious based on historical fraud patterns.
    It helps a fraud review team decide whether a payment needs closer inspection.

    - Higher scores mean greater risk
    - A score above the threshold is treated as suspicious
    - This is a decision-support tool, not an automatic final judgment
    """
)

with st.sidebar:
    st.header("Controls")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    st.markdown("---")
    st.caption("This app calls the FastAPI service and scores transaction data in real time.")

    if st.button("Refresh service status"):
        st.rerun()

col_left, col_middle, col_right = st.columns(3)

with col_left:
    st.markdown("### Model Status")
    try:
        metadata = _get_json(f"{api_url}/metadata")
        st.success("API connected")
        st.json(metadata)
    except Exception as exc:
        st.error(f"Could not reach the API: {exc}")

with col_middle:
    st.markdown("### Health Check")
    try:
        health = _get_json(f"{api_url}/health")
        st.json(health)
    except Exception as exc:
        st.error(str(exc))

with col_right:
    st.markdown("### Decision Policy")
    st.metric("Default threshold", "0.5")
    st.markdown(
        "A score above this threshold is treated as likely fraud. "
        "Teams can raise or lower it depending on how strict they want the review process to be."
    )

st.markdown("---")

st.subheader("Single transaction prediction")

sample_col, input_col = st.columns([1, 3])
with sample_col:
    if st.button("Load sample transaction"):
        st.session_state["sample_payload"] = json.dumps(SAMPLE_TRANSACTION, indent=2)
    if st.button("Clear input"):
        st.session_state.pop("sample_payload", None)

with input_col:
    raw_json = st.text_area(
        "Transaction payload",
        height=220,
        value=st.session_state.get("sample_payload", json.dumps(SAMPLE_TRANSACTION, indent=2)),
    )

if st.button("Predict single transaction"):
    try:
        payload = json.loads(raw_json) if raw_json.strip() else {}
        data = _post_json(f"{api_url}/predict", {"transaction": payload})

        prediction = data["predictions"][0]
        probability = float(prediction["fraud_probability"])
        decision = "FRAUD LIKELY" if prediction["is_fraud"] else "LOW RISK"

        if probability >= 0.5:
            risk_label = "High risk"
            action_text = "This transaction should be reviewed manually or flagged for additional checks."
        elif probability >= 0.2:
            risk_label = "Moderate risk"
            action_text = "This transaction may deserve extra review, but it is not automatically treated as fraud."
        else:
            risk_label = "Low risk"
            action_text = "This transaction looks consistent with normal customer behavior."

        st.markdown("### Result")
        st.metric("Fraud probability", f"{probability:.4f}")
        _render_risk_result(probability, risk_label, decision)
        st.info(action_text)
        st.json(data)
    except requests.HTTPError as exc:
        st.error(f"HTTP error: {exc}")
        try:
            st.json(exc.response.json())
        except Exception:
            pass
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.markdown("---")

st.subheader("Batch prediction from CSV")

st.caption("Upload a CSV containing one transaction per row. Columns should match the model features used during training.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Run batch prediction"):
            data = _post_json(f"{api_url}/predict", {"transactions": df.to_dict(orient="records")})
            st.json(data)
    except Exception as exc:
        st.error(f"Batch scoring failed: {exc}")

st.markdown("---")

st.markdown(
    "### Why this matters\n"
    "Fraud detection is a high-stakes problem because false alarms can block legitimate customers, "
    "while missed fraud can cause financial loss. This app shows how a model can estimate risk and help teams focus their review effort."
)
