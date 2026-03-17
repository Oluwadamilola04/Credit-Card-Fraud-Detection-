# Fraud Detection (End To End: Train, Serve, Monitor)

This project takes a fraud-detection model built in a notebook and turns it into a small, runnable system you can demo and deploy locally. In my workflow, I start with the notebook, save a single artifact, then wrap it with an API, a UI, and basic monitoring.

- Train and evaluate a model on `creditcard.csv`
- Save a deployable artifact (`.pkl`)
- Serve predictions through a FastAPI backend
- Provide a Streamlit UI for manual testing and batch scoring
- Log predictions and feedback labels for monitoring
- Run basic drift checks and export feature importance for sanity checks

## What Problem This Solves

Fraud detection is a highly imbalanced classification problem, so "accuracy" is usually misleading. What you typically want is:

- a **risk score** (fraud probability) per transaction
- a **decision policy** (threshold) you can tune to match operational goals (precision/recall tradeoff)
- a **serving layer** so the model works outside Jupyter
- a **monitoring loop** so you can detect drift and decide when to retrain

## Repo Layout

- `fraud.ipynb`: EDA + preprocessing + training + evaluation (produces the saved artifact)
- `creditcard.csv`: dataset used for training and baseline generation
- `fraud_detection_pipeline.pkl`: preferred deployable artifact (loaded by the API by default)
- `best_fraud_model.pkl`: fallback artifact if the pipeline is not present
- `api/`: FastAPI service (serves predictions)
  - `api/main.py`: endpoints (`/predict`, `/metadata`, `/health`, `/feedback`)
  - `api/schemas.py`: request/response schema (API contract)
  - `api/model.py`: model loading + prediction helpers
  - `api/logging_utils.py`: JSONL logging helpers
- `ui/app.py`: the Streamlit UI that calls the API
- `scripts/`: one-off utilities for monitoring/explainability outputs
  - `scripts/export_baseline_from_csv.py`: generates `artifacts/baseline.json`
  - `scripts/explain.py`: exports `artifacts/feature_importance.csv`
- `monitoring/`:
  - `monitoring/baseline.py`: baseline builder used by the scripts
  - `monitoring/drift.py`: PSI-based drift report
- `DEPLOYMENT.md`: commands and a runbook-style playbook
- `MODEL_CARD.md`: fill-in model documentation (metrics, threshold rationale, limitations)
- `logs/`: runtime logs and JSONL monitoring logs
- `artifacts/`: generated monitoring/explainability outputs

## Quickstart (Start To Finish)

### 1) Environment Setup

Use a Python environment that has (at minimum): `pandas`, `numpy`, `scikit-learn`, `xgboost`, `fastapi`, `uvicorn`, `streamlit`, `requests`. (I tend to use conda for this kind of project, but any environment manager works.)

If you use conda, this is a reasonable starting point:

```powershell
# Example only: create and activate an environment
conda create -n fraud-detect python=3.12 -y
conda activate fraud-detect
pip install -r requirements.txt
```

Note: `requirements.txt` in this repo was generated from an existing environment and may include platform-specific entries. If installation fails, install the key packages listed above instead.

### 2) Train The Model (Notebook)

Open and run:

- `fraud.ipynb`

That notebook is responsible for EDA, preprocessing, training, evaluation, and saving a model artifact.

Expected outputs in the project root:

- `fraud_detection_pipeline.pkl` (preferred)
- `best_fraud_model.pkl` (fallback)

### 3) Start The API (FastAPI)

Start the backend with:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Verify it is healthy:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Check what it loaded (artifact path, threshold, expected features):

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metadata
```

### 4) Start The UI (Streamlit)

Point the UI at the API and run it:

```powershell
$env:API_URL="http://127.0.0.1:8000"
streamlit run ui/app.py
```

Open the printed URL (typically `http://127.0.0.1:8501`) and test:

- a single prediction (paste JSON)
- a batch prediction (upload CSV)

## API Contract (High Level)

### `POST /predict`

`/predict` accepts either:

- a single transaction in `transaction`
- a batch in `transactions`

Each transaction is a JSON object mapping `feature_name -> numeric_value`.

PowerShell example (single):

```powershell
$tx = @{ V1 = 0.0; V2 = 0.0; V3 = 0.0; V4 = 0.0; V5 = 0.0; V6 = 0.0; V7 = 0.0; V8 = 0.0; V9 = 0.0; V10 = 0.0;
         V11 = 0.0; V12 = 0.0; V13 = 0.0; V14 = 0.0; V15 = 0.0; V16 = 0.0; V17 = 0.0; V18 = 0.0; V19 = 0.0; V20 = 0.0;
         V21 = 0.0; V22 = 0.0; V23 = 0.0; V24 = 0.0; V25 = 0.0; V26 = 0.0; V27 = 0.0; V28 = 0.0; Amount = 0.0 }
$body = @{ transaction = $tx } | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $body
```

The response includes:

- `fraud_probability` (0..1)
- `is_fraud` (thresholded decision)
- `request_id` (useful for debugging and later feedback)

### `POST /feedback`

When you later learn the true outcome (ground truth label), send it to `/feedback`. This appends to `logs/feedback.jsonl`.

```powershell
$body = @{ items = @(@{ request_id = "YOUR_REQUEST_ID"; label = 1; metadata = @{ source = "manual_review" } }) } | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/feedback -ContentType "application/json" -Body $body
```

## Configuration

Environment variables supported by the API:

- `MODEL_PATH`: path to the `.pkl` artifact (default: `fraud_detection_pipeline.pkl`, fallback: `best_fraud_model.pkl`)
- `FRAUD_THRESHOLD`: decision threshold (default: `0.5`)
- `PREDICTIONS_LOG`: predictions JSONL path (default: `logs/predictions.jsonl`)
- `FEEDBACK_LOG`: feedback JSONL path (default: `logs/feedback.jsonl`)
- `LOG_FEATURES`: if truthy, logs raw features instead of feature hashes (default: off)

## Monitoring And Drift

### 1) Generate Baseline

To monitor drift, first create a baseline snapshot from the training CSV (writes `artifacts/baseline.json`). In my workflow I do this once per model version (right after training).

```powershell
python scripts/export_baseline_from_csv.py --csv creditcard.csv --out artifacts/baseline.json --model fraud_detection_pipeline.pkl
```

### 2) Generate Drift Report

After the API has produced predictions in `logs/predictions.jsonl`, generate a drift report:

```powershell
python monitoring/drift.py --baseline artifacts/baseline.json --predictions-log logs/predictions.jsonl --out artifacts/drift_report.json
```

Notes:

- Drift is more meaningful after you have a reasonable number of predictions (not just a handful).
- If `LOG_FEATURES` is off (default), the drift tool can still report score drift if the baseline includes stored scores.

## Explainability (Sanity Checks)

Export feature importance:

```powershell
python scripts/explain.py --model fraud_detection_pipeline.pkl --csv creditcard.csv --out artifacts/feature_importance.csv
```

## Suggested Demo (GitHub Screenshots)

- `http://127.0.0.1:8000/docs` (auto-generated API docs)
- the Streamlit UI making a prediction
- `artifacts/feature_importance.csv` (top features)
- `artifacts/drift_report.json` (monitoring output)
- `MODEL_CARD.md` filled with final metrics and threshold rationale

## Limitations And Notes

- This is a demo-style deployment intended for local use and learning.
- For production you would typically add: authentication, rate limiting, structured logs/metrics, and CI/CD.
- Threshold selection should be driven by business cost tradeoffs (false positives vs missed fraud), not by a default.
