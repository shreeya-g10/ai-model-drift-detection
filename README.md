# ai-model-drift-detection

MLOps project for monitoring model performance and drift detection.

## Project Goal

Build an end-to-end pipeline where:
- A spam detection model is served through FastAPI
- Incoming requests are logged and monitored
- Drift/performance changes are detected
- Alerts and retraining triggers can be initiated

## API Contract (Frozen)

### `POST /predict`

Request:

```json
{
  "text": "sample input"
}
```

Response:

```json
{
  "prediction": "spam",
  "confidence": 0.91
}
```

### `GET /metrics-data`

Response contains:
- `total_requests`
- `prediction_counts`
- `logs` (stored prediction logs)

## Team Ownership

- Shreeya: `model/`, `backend/`
- Trisha: `monitoring/`, `drift/`
- Vaidehi: `.github/`, `Dockerfile`

Do not edit files owned by another member unless discussed in a PR.

## Branch & PR Rules

- Branch naming:
  - `feature/shreeya-*`
  - `feature/trisha-*`
  - `feature/vaidehi-*`
- All changes go through Pull Requests
- Keep API format fixed as documented above
- Use small, reviewable PRs with clear scope

## Integration Checklist

- [x] Model training artifacts created in `model/`
- [x] API exposes `POST /predict` and `GET /metrics-data`
- [x] Predictions logged to SQLite
- [x] Prometheus metrics available and scrapeable
- [x] Grafana dashboard configured
- [x] Drift detector flags distribution shift
- [x] Docker image builds and runs
- [x] GitHub Actions pipeline passes

## Local Runbook (Clone to Demo)

1. Clone and enter project:
   - `git clone https://github.com/shreeya-g10/ai-model-drift-detection.git`
   - `cd ai-model-drift-detection`
2. Create virtual environment and install dependencies:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
   - `pip install -r requirements.txt`
3. Train model artifact:
   - `python model/train.py`
4. Start API (use the venv’s Python so deps load; avoids Homebrew `uvicorn` without FastAPI):
   - `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`
5. Test API:
   - `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"win cash now"}'`
   - `curl http://localhost:8000/metrics-data`

### Tests (`pytest`)

Run everything from **inside** `ai-model-drift-detection` (not the parent `devops` folder), with **this** project’s virtualenv active:

```bash
cd ai-model-drift-detection
source .venv/bin/activate
pip install -r requirements.txt
python model/train.py
python -m pytest -q
```

Use `python -m pytest` so tests use the same interpreter as `pip install` (avoids a global/Homebrew `pytest` without FastAPI).

If you still see `ModuleNotFoundError: No module named 'fastapi'`, check `which python` — it must point under `ai-model-drift-detection/.venv/`.

## Monitoring Setup

- Prometheus config: `monitoring/prometheus.yml`
- Grafana dashboard template: `monitoring/grafana_dashboard.json`
- Optional local monitoring stack:
  - `cd monitoring`
  - `docker compose -f docker-compose.monitoring.yml up -d`
- API exposes Prometheus metrics at `GET /metrics`.

### Grafana datasource (`http://prometheus:9090` fails)

That URL only works when **Grafana runs inside Docker** and shares a **Compose network** with the **Prometheus** service (name `prometheus`). It will not open in your normal browser.

1. Recreate the stack so both containers share the compose network:

   ```bash
   cd monitoring
   docker compose -f docker-compose.monitoring.yml down
   docker compose -f docker-compose.monitoring.yml up -d
   ```

2. Prove Grafana can reach Prometheus (should print `Prometheus Server is Healthy.`):

   ```bash
   docker exec ml-grafana curl -sf http://prometheus:9090/-/healthy
   ```

   If `curl` is missing in the container, try: `docker exec ml-grafana wget -qO- http://prometheus:9090/-/healthy`

   If this fails, Grafana was started outside this compose file or uses another network — use **`http://host.docker.internal:9090`** instead (Prometheus on the Mac), or put both services in this compose file only.

3. In Grafana → Connections → Data sources → Prometheus, set URL **`http://prometheus:9090`**, Save & test.

## Drift Detection

- Script: `drift/check_drift.py`
- Example:
  - `python drift/check_drift.py --window-size 50 --threshold 0.2`
- Alert behavior:
  - Prints `Drift detected!` when class distribution shift exceeds threshold.

## CI/CD

- Workflow file: `.github/workflows/ci-cd.yml`
- Pipeline stages:
  - dependency install
  - model training
  - tests
  - docker build
  - docker push on `main` pushes
- Required GitHub Secrets:
  - `DOCKERHUB_USERNAME`
  - `DOCKERHUB_TOKEN`
