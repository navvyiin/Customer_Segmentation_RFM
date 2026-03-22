# Customer Segmentation & Intelligence Platform

An end-to-end customer analytics platform built in Python. Goes beyond standard RFM analysis by combining probabilistic lifetime value modelling, churn prediction, causal uplift analysis, customer embeddings, and real-time inference, all packaged with an interactive dashboard and a REST API.

---

## What It Does

- Segments customers using RFM scoring and K-Means clustering
- Forecasts customer lifetime value probabilistically using BG/NBD and Gamma-Gamma models
- Predicts churn risk with SHAP-based feature-level explanations
- Measures marketing treatment effects using causal uplift modelling
- Finds similar customers using Customer2Vec-style embeddings and FAISS similarity search
- Serves real-time predictions through a FastAPI inference service
- Presents all insights in an interactive Streamlit dashboard

---

## Architecture

The repository is structured as a modular ML system, not a single notebook.

```
Customer_Segmentation_RFM/
├── app.py                  # Streamlit dashboard
├── fastapi_scorer.py       # FastAPI inference service
├── producer_sim.py         # Real-time event simulator
├── rfm_utils.py            # RFM computation
├── cltv_models.py          # CLTV modelling
├── uplift.py               # Causal uplift
├── embeddings.py           # Customer embeddings and FAISS
├── explainability.py       # SHAP explanations
├── campaign_planner.py     # Campaign targeting logic
├── models/                 # Saved model artefacts
├── docs/
│   ├── architecture.md
│   └── model_card.md
├── requirements.txt
├── contributing.md
└── code_of_conduct.md
```

---

## Getting Started

**Requirements:** Python 3.8 or above

```bash
git clone https://github.com/navvyiin/Customer_Segmentation_RFM.git
cd Customer_Segmentation_RFM
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

---

## Running the Project

**Dashboard**

```bash
streamlit run app.py
```

Explore RFM segments, CLTV and churn predictions, SHAP plots, and customer similarity from the sidebar.

**Inference API**

```bash
uvicorn fastapi_scorer:app --reload --port 8001
```

Available endpoints: `POST /predict_churn`, `POST /predict_cltv`, `POST /uplift_score`, `POST /similar_customers`

**Event Simulation (optional)**

```bash
python producer_sim.py
```

Simulates a stream of customer events for testing real-time scoring pipelines.

---

## Tech Stack

`Python` `Pandas` `Scikit-learn` `Lifetimes` `PyTorch` `SHAP` `FAISS` `FastAPI` `Streamlit`

---

## Notes

- Built for experimentation, research, and ML system design demonstrations
- Processing speed depends on dataset size and hardware
- Not designed as a plug-and-play production SaaS product

---

## Roadmap

- Dockerised deployment
- Model monitoring and drift detection
- Feature store integration
- CI/CD and automated testing
- Cloud-native deployment examples

---

## Contributing

Contributions and ideas are welcome. Please read `contributing.md` before submitting a pull request.

---

## License

MIT License. © 2026 navvyiin
