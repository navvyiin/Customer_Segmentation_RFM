from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
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