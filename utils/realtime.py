FASTAPI_SCORER = """\
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

with open("models/churn_model.pkl", "rb") as f:
    churn_model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


class CustomerEvent(BaseModel):
    customer_id: str
    recency: float
    frequency: float
    monetary: float


@app.post("/score_churn")
def score_churn(event: CustomerEvent):
    x = np.array([[np.log1p(event.recency),
                   np.log1p(event.frequency),
                   np.log1p(event.monetary)]])
    x_scaled = scaler.transform(x)
    prob = float(churn_model.predict_proba(x_scaled)[:, 1])
    return {"customer_id": event.customer_id, "churn_prob": prob}
"""

PRODUCER_SIM = """\
import requests
import time
import random

URL = "http://localhost:8000/score_churn"

def simulate_event():
    return {
        "customer_id": f"C{random.randint(10000, 99999)}",
        "recency": random.uniform(1, 365),
        "frequency": random.uniform(1, 20),
        "monetary": random.uniform(10, 1000),
    }

if __name__ == "__main__":
    while True:
        event = simulate_event()
        r = requests.post(URL, json=event)
        print("Sent:", event, "Got:", r.json())
        time.sleep(1)
"""