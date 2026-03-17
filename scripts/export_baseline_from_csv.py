from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure repo root is on sys.path when running `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitoring.baseline import build_baseline, write_baseline


def _load_model(path: str):
    try:
        import joblib  # type: ignore

        return joblib.load(path)
    except Exception:
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="creditcard.csv")
    ap.add_argument("--label-col", default="Class")
    ap.add_argument("--out", default="artifacts/baseline.json")
    ap.add_argument("--drop-cols", default="Time")
    ap.add_argument(
        "--model",
        default="",
        help="Optional model/pipeline path to also store baseline score distribution.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.label_col in df.columns:
        df = df.drop(columns=[args.label_col])

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    scores = None
    if args.model:
        model = _load_model(args.model)
        # Use the same prediction helper as the API so artifact quirks are handled consistently.
        from api.model import predict_proba

        scores = predict_proba(model, df)

    baseline = build_baseline(df, scores=scores)
    write_baseline(args.out, baseline)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
