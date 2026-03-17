from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index for numeric arrays.
    """

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if cuts.size < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=cuts)
    actual_counts, _ = np.histogram(actual, bins=cuts)

    expected_perc = expected_counts / max(expected_counts.sum(), 1)
    actual_perc = actual_counts / max(actual_counts.sum(), 1)

    # Avoid div-by-zero; tiny smoothing
    eps = 1e-6
    expected_perc = np.clip(expected_perc, eps, 1)
    actual_perc = np.clip(actual_perc, eps, 1)

    return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="artifacts/baseline.json")
    ap.add_argument("--predictions-log", default="logs/predictions.jsonl")
    ap.add_argument("--out", default="artifacts/drift_report.json")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    pred_path = Path(args.predictions_log)
    out_path = Path(args.out)

    if not baseline_path.exists():
        raise SystemExit(f"Baseline not found: {baseline_path}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(pred_path)
    if not rows:
        raise SystemExit(f"No prediction logs found at {pred_path}")

    # We only log raw features if LOG_FEATURES=true. If not present, drift checks are limited.
    feature_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for r in rows:
        scores.extend([float(x) for x in r.get("fraud_probabilities", [])])
        feats = r.get("features")
        if isinstance(feats, list):
            feature_rows.extend(feats)

    report: dict[str, Any] = {
        "n_prediction_records": len(rows),
        "n_scored_items": len(scores),
        "n_feature_rows": len(feature_rows),
        "score_psi": None,
        "feature_psi": {},
    }

    if scores and "scores" in baseline:
        report["score_psi"] = psi(np.array(baseline["scores"], dtype=float), np.array(scores, dtype=float))

    if feature_rows and "features" in baseline:
        df = pd.DataFrame(feature_rows)
        for feat, expected_values in baseline["features"].items():
            if feat not in df.columns:
                continue
            actual_values = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float)
            report["feature_psi"][feat] = psi(np.array(expected_values, dtype=float), actual_values)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

