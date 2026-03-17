# Model Card: Fraud Detection

## Overview
- **Problem:** Credit card fraud detection (binary classification)
- **Intended use:** Assist fraud review/automation based on a calibrated risk score
- **Out of scope:** Any use where false positives/negatives are not tolerable (human review recommended)

## Data
- **Dataset:** `creditcard.csv` (features `V1..V28`, `Time`, `Amount`, label `Class`)
- **Label meaning:** `Class=1` fraud, `Class=0` non-fraud
- **Notes:** Features `V1..V28` are anonymized (PCA-like); `Amount` is transaction amount

## Training
- **Train/val split:** (fill in)
- **Imbalance handling:** (fill in, e.g., class weights / SMOTE applied to training split only)
- **Preprocessing:** (fill in, e.g., drop `Time`, scale `Amount`, etc.)
- **Model:** (fill in, e.g., XGBoost / Random Forest / Logistic Regression)

## Metrics
- **ROC-AUC:** (fill in)
- **PR-AUC (Average Precision):** (fill in)
- **Confusion matrix at chosen threshold:** (fill in)
- **Calibration:** (fill in: method + plot results)

## Decision Policy
- **Threshold:** `FRAUD_THRESHOLD` (default `0.5`)
- **Rationale:** (fill in: cost tradeoff, target precision/recall, operational constraints)

## Explainability / Sanity Checks
- **Top features:** see `artifacts/feature_importance.csv`
- **Notes:** (fill in: known spurious patterns, stability, etc.)

## Deployment
- **API:** FastAPI service (`api/main.py`)
- **UI:** Streamlit app (`ui/app.py`)
- **Artifacts:** serialized model/pipeline (`fraud_detection_pipeline.pkl` or `best_fraud_model.pkl`)

## Monitoring
- **Prediction logging:** JSONL at `logs/predictions.jsonl`
- **Feedback logging:** JSONL at `logs/feedback.jsonl`
- **Drift checks:** `python -m monitoring.drift --baseline artifacts/baseline.json`
- **Retraining triggers:** (fill in: drift threshold, cadence, label availability)

## Limitations
- Extremely imbalanced data; accuracy is not meaningful.
- Performance may degrade under distribution shift.
- Threshold is business-dependent; default is not production-ready.

