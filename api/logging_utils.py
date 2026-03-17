from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def predictions_log_path() -> Path:
    return Path(os.getenv("PREDICTIONS_LOG", "logs/predictions.jsonl"))


def feedback_log_path() -> Path:
    return Path(os.getenv("FEEDBACK_LOG", "logs/feedback.jsonl"))


def should_log_features() -> bool:
    raw = os.getenv("LOG_FEATURES", "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}

