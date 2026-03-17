from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

try:
    # Pydantic v2
    from pydantic import model_validator  # type: ignore
except Exception:  # pragma: no cover
    model_validator = None  # type: ignore
    # Pydantic v1 fallback
    from pydantic import root_validator  # type: ignore


class PredictRequest(BaseModel):
    """
    Accept either:
      - a single transaction dict in `transaction`
      - a batch in `transactions`
    Each transaction is a mapping of feature name -> numeric value.
    """

    transaction: dict[str, float] | None = None
    transactions: list[dict[str, float]] | None = None

    if model_validator is not None:

        @model_validator(mode="after")  # type: ignore[misc]
        def _validate_one_of(self) -> "PredictRequest":
            if self.transaction is None and self.transactions is None:
                raise ValueError("Provide `transaction` or `transactions`.")
            if self.transaction is not None and self.transactions is not None:
                raise ValueError("Provide only one of `transaction` or `transactions`.")
            return self

    else:

        @root_validator  # type: ignore[misc]
        def _validate_one_of_v1(cls, values: dict[str, Any]) -> dict[str, Any]:
            tx = values.get("transaction")
            txs = values.get("transactions")
            if tx is None and txs is None:
                raise ValueError("Provide `transaction` or `transactions`.")
            if tx is not None and txs is not None:
                raise ValueError("Provide only one of `transaction` or `transactions`.")
            return values


class PredictItem(BaseModel):
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_fraud: bool


class PredictResponse(BaseModel):
    request_id: str
    model_path: str
    threshold: float
    created_at: datetime
    predictions: list[PredictItem]


class FeedbackItem(BaseModel):
    request_id: str
    label: int = Field(..., ge=0, le=1)
    metadata: dict[str, Any] | None = None


class FeedbackRequest(BaseModel):
    items: list[FeedbackItem]


class FeedbackResponse(BaseModel):
    accepted: int
