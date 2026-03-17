from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_model(path: str) -> Any:
    try:
        import joblib  # type: ignore

        return joblib.load(path)
    except Exception:
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)


def _get_estimator(model: Any) -> Any:
    # Pipeline: grab last step; otherwise return model as-is.
    steps = getattr(model, "steps", None)
    if steps:
        return steps[-1][1]
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fraud_detection_pipeline.pkl")
    ap.add_argument("--csv", default="creditcard.csv")
    ap.add_argument("--label-col", default="Class")
    ap.add_argument("--drop-cols", default="Time")
    ap.add_argument("--out", default="artifacts/feature_importance.csv")
    args = ap.parse_args()

    model = _load_model(args.model)
    estimator = _get_estimator(model)

    df = pd.read_csv(args.csv)
    if args.label_col in df.columns:
        y = df[args.label_col].to_numpy()
        X = df.drop(columns=[args.label_col])
    else:
        raise SystemExit(f"Label column not found: {args.label_col}")

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    for c in drop_cols:
        if c in X.columns:
            X = X.drop(columns=[c])

    # Prefer native feature_importances_ when available.
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        feature_names = list(getattr(estimator, "feature_names_in_", X.columns))
    else:
        # Fall back to permutation importance on a small sample to keep runtime reasonable.
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # If the artifact is a pipeline, it can handle preprocessing.
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring="average_precision",
        )
        importances = np.asarray(result.importances_mean, dtype=float)
        feature_names = list(X_test.columns)

    rows = sorted(
        [{"feature": str(f), "importance": float(v)} for f, v in zip(feature_names, importances)],
        key=lambda r: r["importance"],
        reverse=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "importance"])
        w.writeheader()
        w.writerows(rows)

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

