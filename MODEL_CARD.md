# Model Card: Fraud Detection

## Overview
- **Problem:** Credit card fraud detection as a binary classification problem.
- **Intended use:** Assist fraud review teams by estimating the probability that a transaction is fraudulent.
- **Out of scope:** Fully autonomous fraud blocking without human review. This is a decision-support tool, not a final authority.

## Data
- **Dataset:** `creditcard.csv`
- **Feature structure:** anonymized transaction features `V1` through `V28`, transaction `Time`, transaction `Amount`, and target label `Class`
- **Label meaning:** `Class=1` indicates fraud; `Class=0` indicates non-fraud
- **Notes:** The feature set is anonymized and highly imbalanced, which is common in fraud detection.

## Training
- **Train/test split:** modeled as a stratified classification task to preserve the minority class proportion in validation
- **Imbalance handling:** the notebook focuses on threshold tuning and model selection rather than heavy oversampling; the model is selected based on ranking performance under class imbalance
- **Preprocessing:** feature engineering and selected preprocessing steps are handled in the training pipeline, with a model artifact saved for reuse in serving
- **Model family:** tuned XGBoost classifier, selected as the strongest performer among evaluated models

## Metrics
- **ROC-AUC:** approximately `0.9758`
- **Average Precision / PR-AUC:** strong performance relative to base rate, with precision-recall tuning used to select threshold
- **Selected threshold:** around `0.9742` as the best F1-driven operating point in the notebook tuning experiment
- **Representative metrics at tuned threshold:**
  - Precision: `0.9518`
  - Recall: `0.8061`
  - F1-score: `0.8729`

These metrics should be interpreted in the context of extreme class imbalance; the chosen threshold is designed to balance fraud detection with false-positive risk.

## Decision Policy
- **Default API threshold:** `0.5`
- **Operational threshold explored during model tuning:** approximately `0.9742` for the best F1 tradeoff in the notebook
- **Rationale:** in fraud settings, a threshold should reflect business cost: false positives can frustrate customers, while false negatives can allow real fraud. The project demonstrates that threshold selection matters as much as model choice.

## Explainability / Sanity Checks
- **Top features:** see `artifacts/feature_importance.csv`
- **Notes:** the model is meant to produce a risk signal, not a guaranteed causal explanation of fraud. Feature importance is a useful sanity check, not a final legal or operational explanation.

## Deployment
- **API:** FastAPI service in `api/main.py`
- **UI:** Streamlit app in `ui/app.py`
- **Artifacts:** serialized sklearn pipeline or model object saved in the project root, such as `fraud_detection_pipeline.pkl`

## Monitoring
- **Prediction logging:** JSONL at `logs/predictions.jsonl`
- **Feedback logging:** JSONL at `logs/feedback.jsonl`
- **Drift checks:** PSI-based drift analysis using `monitoring/drift.py`
- **Retraining triggers:** ideally triggered by sustained drift, threshold miss rate changes, or newly labeled fraud observations

## Limitations
- Extremely imbalanced data means accuracy is not a meaningful headline metric.
- The dataset is anonymized, which reduces interpretability for business stakeholders.
- Model performance may degrade under concept drift or changing fraud patterns.
- The default threshold of `0.5` is not universally appropriate; it should be tuned to the business objective.

## Summary
This project is designed as an end-to-end fraud detection demo: it trains a competitive model, serves predictions through an API, exposes the model through a UI, and demonstrates the monitoring discipline expected in real ML systems.

