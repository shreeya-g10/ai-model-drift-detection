import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from prometheus_client import Counter
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "model.pkl"
DB_PATH = ROOT / "backend" / "predictions.db"

app = FastAPI(title="AI Model Monitoring API")
# Must register middleware at import time; startup handlers run too late for add_middleware.
Instrumentator().instrument(app).expose(app)

PREDICTION_COUNTER = Counter(
    "prediction_count_total",
    "Total predictions by class",
    ["prediction"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text for spam prediction")


def _create_connection():
    """Open SQLite without going through get_connection() (avoids recursion with init_db)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH.resolve()), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _create_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_connection():
    # Ensure schema exists on every use so logging works even if ASGI startup did not run.
    init_db()
    return _create_connection()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run `python model/train.py` first."
        )
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def get_model():
    if not hasattr(app.state, "model"):
        app.state.model = load_model()
    return app.state.model


@app.on_event("startup")
def startup_event():
    init_db()
    app.state.model = load_model()


@app.post("/predict")
def predict(payload: PredictRequest):
    model = get_model()
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = round(float(max(probabilities)), 4)
    PREDICTION_COUNTER.labels(prediction=str(prediction)).inc()

    timestamp = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO prediction_logs (input_text, prediction, confidence, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (text, str(prediction), confidence, timestamp),
        )
        conn.commit()

    return {"prediction": str(prediction), "confidence": confidence}


@app.get("/metrics-data")
def metrics_data(limit: int = Query(100, ge=1, le=500)):
    with get_connection() as conn:
        total_requests = conn.execute("SELECT COUNT(*) AS count FROM prediction_logs").fetchone()[
            "count"
        ]

        prediction_rows = conn.execute(
            "SELECT prediction, COUNT(*) AS count FROM prediction_logs GROUP BY prediction"
        ).fetchall()
        prediction_counts = {row["prediction"]: row["count"] for row in prediction_rows}

        logs = conn.execute(
            """
            SELECT input_text, prediction, confidence, timestamp
            FROM prediction_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    serialized_logs = [
        {
            "input_text": row["input_text"],
            "prediction": row["prediction"],
            "confidence": row["confidence"],
            "timestamp": row["timestamp"],
        }
        for row in logs
    ]

    return {
        "total_requests": total_requests,
        "prediction_counts": prediction_counts,
        "logs": serialized_logs,
    }