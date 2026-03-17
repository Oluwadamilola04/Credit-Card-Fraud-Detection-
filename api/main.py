from __future__ import annotations

import hashlib
import logging
import os
import traceback
import uuid
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.logging_utils import (
    append_jsonl,
    feedback_log_path,
    predictions_log_path,
    should_log_features,
    utc_now,
)
from api.model import get_loaded_model, predict_proba, required_feature_names
from api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    PredictItem,
    PredictRequest,
    PredictResponse,
)


app = FastAPI(title="Fraud Detection API", version="0.1.0")  # deployed service
logger = logging.getLogger("fraud_api")


def _hash_features(tx: dict[str, float]) -> str:
    # Stable-ish fingerprint without storing raw features.
    items = sorted((k, float(v)) for k, v in tx.items())
    payload = "|".join(f"{k}={v:.12g}" for k, v in items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _coerce_transactions(payload: PredictRequest) -> list[dict[str, float]]:
    if payload.transaction is not None:
        return [payload.transaction]
    assert payload.transactions is not None
    return payload.transactions


def _to_dataframe(txs: list[dict[str, float]], feature_order: list[str] | None) -> pd.DataFrame:
    df = pd.DataFrame(txs)

    # Optional convenience: if client sends Time but model doesn't need it.
    if "Time" in df.columns and feature_order is not None and "Time" not in feature_order:
        df = df.drop(columns=["Time"])

    if feature_order is not None:
        missing = [c for c in feature_order if c not in df.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing required features: {missing}")
        df = df[feature_order]

    # Ensure numeric
    try:
        df = df.apply(pd.to_numeric)
    except Exception:
        raise HTTPException(status_code=422, detail="All feature values must be numeric.")

    return df


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    try:
        loaded = get_loaded_model()
        feature_order = required_feature_names(loaded.model)
        return {
            "model_path": loaded.model_path,
            "threshold": loaded.threshold,
            "expected_features": feature_order,
        }
    except Exception as e:
        return {"model_error": str(e)}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    request_id = uuid.uuid4().hex
    created_at = utc_now()

    try:
        loaded = get_loaded_model()
        feature_order = required_feature_names(loaded.model)

        txs = _coerce_transactions(payload)
        df = _to_dataframe(txs, feature_order)

        probs = predict_proba(loaded.model, df)
        items = [
            PredictItem(
                fraud_probability=p,
                is_fraud=(p >= loaded.threshold),
            )
            for p in probs
        ]

        # Monitoring log (JSONL)
        log_record: dict[str, Any] = {
            "created_at": created_at.isoformat(),
            "request_id": request_id,
            "model_path": loaded.model_path,
            "threshold": loaded.threshold,
            "n_items": len(txs),
            "fraud_probabilities": probs,
            "decisions": [int(p >= loaded.threshold) for p in probs],
        }
        if should_log_features():
            log_record["features"] = txs
        else:
            log_record["feature_hashes"] = [_hash_features(tx) for tx in txs]

        append_jsonl(predictions_log_path(), log_record)

        return PredictResponse(
            request_id=request_id,
            model_path=loaded.model_path,
            threshold=loaded.threshold,
            created_at=created_at,
            predictions=items,
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        # Ensure we capture the root cause in process logs even if logging isn't configured.
        traceback.print_exc()
        logger.exception("Prediction failed (request_id=%s)", request_id)
        # Avoid leaking internals; provide request_id for debugging.
        raise HTTPException(status_code=500, detail=f"Prediction failed (request_id={request_id}).") from e


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest) -> FeedbackResponse:
    accepted = 0
    now = utc_now().isoformat()

    for item in payload.items:
        record: dict[str, Any] = {
            "created_at": now,
            "request_id": item.request_id,
            "label": int(item.label),
            "metadata": item.metadata,
        }
        append_jsonl(feedback_log_path(), record)
        accepted += 1

    return FeedbackResponse(accepted=accepted)


@app.exception_handler(HTTPException)
def http_exception_handler(_: Any, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
