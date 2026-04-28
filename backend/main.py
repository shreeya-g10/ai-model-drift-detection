import os

# Ensure log folder + file exist
os.makedirs("log", exist_ok=True)
if not os.path.exists("log/log.json"):
    open("log/log.json", "w").close()

from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import time

app = FastAPI()

# Load model + vectorizer
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Define input format (JSON)
class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    text = data.text
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]

    # Logging
    log = {
        "text": text,
        "prediction": str(prediction),
        "timestamp": time.time()
    }

    with open("log/log.json", "a") as f:
        f.write(json.dumps(log) + "\n")

    return {"prediction": str(prediction)}


@app.get("/metrics-data")
def metrics():
    try:
        with open("log/log.json") as f:
            logs = f.readlines()
    except:
        logs = []

    return {"total_requests": len(logs)}