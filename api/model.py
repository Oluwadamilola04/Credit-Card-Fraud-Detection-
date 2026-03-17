from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid float for {name}={raw!r}") from e


def _resolve_model_path() -> str:
    # Prefer pipeline artifact if present; fall back to best_model.
    return os.getenv("MODEL_PATH") or (
        "fraud_detection_pipeline.pkl"
        if Path("fraud_detection_pipeline.pkl").exists()
        else "best_fraud_model.pkl"
    )


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    model_path: str
    threshold: float


_lock = threading.Lock()
_cached: LoadedModel | None = None


def get_loaded_model() -> LoadedModel:
    """
    Lazily loads the serialized model/pipeline once per process.
    Supports artifacts saved via joblib or pickle.
    """

    global _cached
    with _lock:
        if _cached is not None:
            return _cached

        model_path = _resolve_model_path()
        threshold = _env_float("FRAUD_THRESHOLD", 0.5)

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Set MODEL_PATH env var or place the artifact in the project root."
            )

        # Try joblib first (common for sklearn), then pickle.
        try:
            import joblib  # type: ignore

            model = joblib.load(model_path)
        except Exception:
            import pickle

            with open(model_path, "rb") as f:
                model = pickle.load(f)

        if not hasattr(model, "predict_proba") and not hasattr(model, "steps"):
            raise TypeError(
                f"Loaded object from {model_path} is not a model/pipeline with predict_proba(). "
                f"Got: {type(model)}"
            )

        _cached = LoadedModel(model=model, model_path=model_path, threshold=threshold)
        return _cached


def expected_feature_names(model: Any) -> list[str] | None:
    """
    Best-effort discovery of expected input features for ordering/validation.
    """

    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        try:
            return [str(x) for x in list(names)]
        except Exception:
            return None

    # Pipeline -> last step might carry feature_names_in_
    steps = getattr(model, "steps", None)
    if steps:
        for _, step in reversed(steps):
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                try:
                    return [str(x) for x in list(names)]
                except Exception:
                    return None

    return None


def required_feature_names(model: Any) -> list[str] | None:
    """
    Returns a "best" required feature list for validation.

    Many artifacts expose incomplete or misleading feature name metadata (especially when
    trained on numpy arrays). If we can infer the expected feature count and it matches the
    common `creditcard.csv` schema, we require that schema.
    """

    default_29 = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    default_30 = ["Time"] + default_29

    names = expected_feature_names(model)
    if names is not None and len(names) >= 5:
        return names

    # If this is a pipeline, infer from final estimator.
    est = model
    steps = getattr(model, "steps", None)
    if steps:
        est = steps[-1][1]

    n = getattr(est, "n_features_in_", None)
    try:
        n_int = int(n) if n is not None else None
    except Exception:
        n_int = None

    if n_int == 29:
        return default_29
    if n_int == 30:
        return default_30

    # If we cannot infer, do not enforce.
    return None


def predict_proba(model: Any, df: Any) -> list[float]:
    """
    Returns fraud probabilities for class 1.
    Requires predict_proba.
    """

    # Special-case: the saved artifact may be a 2-step sklearn Pipeline:
    #   scaler(StandardScaler fit on ['Amount']) -> model(XGBClassifier fit on 29 features)
    # In that case, calling pipeline.predict_proba(df) fails because the scaler was fit on a
    # 1-column DataFrame, but the incoming request contains all features. We replicate the
    # intended preprocessing (scale Amount only) and then call the estimator directly.
    steps = getattr(model, "steps", None)
    if steps and len(steps) == 2:
        scaler = steps[0][1]
        estimator = steps[1][1]
        scaler_cols = getattr(scaler, "feature_names_in_", None)
        est_cols = getattr(estimator, "feature_names_in_", None)
        if scaler_cols is not None and est_cols is not None:
            scaler_cols = [str(x) for x in list(scaler_cols)]
            est_cols = [str(x) for x in list(est_cols)]
            if scaler_cols == ["Amount"] and "Amount" in est_cols:
                try:
                    amount_scaled = scaler.transform(df[["Amount"]])
                    df2 = df.copy()
                    df2["Amount"] = amount_scaled.reshape(-1)
                    df2 = df2[est_cols]
                    proba = estimator.predict_proba(df2)
                    return [float(p[1]) for p in proba]
                except Exception as e:
                    raise TypeError("Failed to run pipeline special-case prediction.") from e

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")

    proba = model.predict_proba(df)
    # sklearn returns shape (n, 2) for binary classification
    try:
        return [float(p[1]) for p in proba]
    except Exception as e:
        raise TypeError("Unexpected predict_proba output shape.") from e
