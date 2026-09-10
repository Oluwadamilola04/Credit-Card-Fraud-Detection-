# Minimal Deployment (API + UI)

## Playbook (Order Of Operations)

### Day 1: Train And Produce Artifacts
1. Run `fraud.ipynb` end-to-end (EDA, training, evaluation).
2. Save a deployable artifact into the project root:
   - Preferred: `fraud_detection_pipeline.pkl` (preprocessing + model)
   - Fallback: `best_fraud_model.pkl` (model only)
3. Decide your operating threshold (business-dependent) and set it for serving:
   - `FRAUD_THRESHOLD` (default `0.5`)

### Day 2: Serve And Smoke Test
1. Start the API.
2. Confirm it is up with `GET /health`.
3. Check `GET /metadata` to confirm:
   - which model file loaded
   - which threshold is active
   - which features are expected (if the artifact exposes them)
4. Start the UI and perform a few manual predictions (single + batch).

### Day 3: Enable Logging And Monitoring
1. Decide what to log:
   - Default: log scores + decisions + feature hashes (safer)
   - Optional: log raw features by setting `LOG_FEATURES=true` (enables feature drift checks)
2. Generate a baseline snapshot for drift checks:
   - `artifacts/baseline.json` from `creditcard.csv`
   - Optional: include baseline score distribution by providing `--model`
3. Run drift reports on a schedule (daily/weekly) and store `artifacts/drift_report.json`.

### Ongoing: Feedback And Retraining
1. When ground-truth becomes available, send labels to `POST /feedback` (writes `logs/feedback.jsonl`).
2. Periodically:
   - review drift report
   - review score distributions and false positive rates at your chosen threshold
   - retrain/tune if drift is high or performance drops

## Artifacts
The API expects a serialized sklearn model/pipeline file in the project root.

- Default: `fraud_detection_pipeline.pkl`
- Fallback: `best_fraud_model.pkl`
- Override: set `MODEL_PATH`

Optional:
- Set `FRAUD_THRESHOLD` (default `0.5`)

## Run API (FastAPI)
From your conda environment:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Endpoints:
- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /feedback`

## Run UI (Streamlit)

```powershell
$env:API_URL="http://127.0.0.1:8000"
streamlit run ui/app.py
```

### Streamlit Community Cloud note

Streamlit Community Cloud runs the Streamlit interface, but it does not automatically start the FastAPI service in `api/main.py`. For a hosted deployment, run the API separately on a service that supports a persistent web process, then set the Streamlit app's `API_URL` secret to that public API URL. The local two-process setup above is still required for local development.

#### Recommended hosted setup

Use two services:

1. **API service**: deploy this repository to a Python web-service host.
   - Build/install command: `pip install -r requirements.txt`
   - Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - Make sure `fraud_detection_pipeline.pkl` is included in the deployed repository, or set `MODEL_PATH` to the deployed artifact path.
   - Copy the public service URL and verify `https://YOUR-API-URL/health` returns `{"status":"ok"}`.
2. **Streamlit Community Cloud**: deploy `ui/app.py` from this repository.
   - In the app's settings, add this secret:

     ```toml
     API_URL = "https://YOUR-API-URL"
     ```

   - The app uses that URL for `/health`, `/metadata`, and `/predict`.

Do not set `API_URL` to `localhost` or `127.0.0.1` in the hosted Streamlit app. Those addresses refer to the Streamlit container itself, not to the separately deployed API. For local development, keep using `http://127.0.0.1:8000` and run both processes.

## Monitoring

### Enable richer drift monitoring
If you are comfortable logging raw features (features are anonymized in this dataset):

```powershell
$env:LOG_FEATURES="true"
```

### Baseline (from training CSV)

```powershell
python scripts/export_baseline_from_csv.py --csv creditcard.csv --out artifacts/baseline.json --model fraud_detection_pipeline.pkl
```

### Drift report

```powershell
python monitoring/drift.py --baseline artifacts/baseline.json --predictions-log logs/predictions.jsonl --out artifacts/drift_report.json
```

## Explainability

```powershell
python scripts/explain.py --model fraud_detection_pipeline.pkl --csv creditcard.csv --out artifacts/feature_importance.csv
```
