from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import app, init_db


def test_predict_and_metrics():
    init_db()
    client = TestClient(app)

    response = client.post("/predict", json={"text": "Win a free prize now"})
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "confidence" in body
    assert isinstance(body["confidence"], float)

    metrics_response = client.get("/metrics-data")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert "total_requests" in metrics
    assert "prediction_counts" in metrics
    assert "logs" in metrics
