from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_baseline(df: pd.DataFrame, scores: np.ndarray | None = None, max_rows: int = 50000) -> dict[str, Any]:
    """
    Stores a capped sample of feature values for PSI-based drift checks.
    Keep it small and non-sensitive; for this dataset features are anonymized.
    """

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    baseline: dict[str, Any] = {"features": {}}
    for c in df.columns:
        series = pd.to_numeric(df[c], errors="coerce")
        values = series.dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        # Cap stored values for file size
        if values.size > 50000:
            values = np.random.default_rng(42).choice(values, size=50000, replace=False)
        baseline["features"][str(c)] = values.astype(float).tolist()

    if scores is not None:
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        if scores.size > 50000:
            scores = np.random.default_rng(42).choice(scores, size=50000, replace=False)
        baseline["scores"] = scores.tolist()

    return baseline


def write_baseline(path: str | Path, baseline: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, ensure_ascii=True), encoding="utf-8")

