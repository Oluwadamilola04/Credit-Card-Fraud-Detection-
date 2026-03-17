from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


st.set_page_config(page_title="Fraud Detection", page_icon="F", layout="wide")
st.title("Fraud Detection UI")

api_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)

col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("API Status")
    if st.button("Check /health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=10)
            st.json(r.json())
        except Exception as e:
            st.error(str(e))

    if st.button("View /metadata"):
        try:
            r = requests.get(f"{api_url}/metadata", timeout=10)
            st.json(r.json())
        except Exception as e:
            st.error(str(e))

with col_b:
    st.subheader("Single Prediction (JSON)")
    st.caption("Paste a JSON object of feature->value. Example: {\"V1\": -1.23, ..., \"Amount\": 12.34}")
    raw = st.text_area("Transaction JSON", height=180, value="{}")
    if st.button("Predict (single)"):
        try:
            tx = json.loads(raw) if raw.strip() else {}
            data = _post_json(f"{api_url}/predict", {"transaction": tx})
            st.json(data)
        except requests.HTTPError as e:
            st.error(f"HTTP error: {e}")
            try:
                st.json(e.response.json())
            except Exception:
                pass
        except Exception as e:
            st.error(str(e))

st.divider()

st.subheader("Batch Prediction (CSV Upload)")
st.caption("Upload a CSV with one row per transaction. Columns should match your model features.")
file = st.file_uploader("CSV file", type=["csv"])
if file is not None:
    try:
        df = pd.read_csv(file)
        st.write("Preview")
        st.dataframe(df.head(10), use_container_width=True)
        if st.button("Predict (batch)"):
            txs = df.to_dict(orient="records")
            data = _post_json(f"{api_url}/predict", {"transactions": txs})
            st.write("Predictions")
            st.json(data)
    except Exception as e:
        st.error(str(e))
