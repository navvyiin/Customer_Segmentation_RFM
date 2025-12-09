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