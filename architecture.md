**Corporate RFM Analytics Suite**
Advanced Segmentation • CLTV Modelling • Causal Uplift • Embeddings • Real-time ML

---

# 1. **High-Level Overview**

The Corporate RFM Analytics Suite is a full-stack customer intelligence platform combining:

* Transactional data processing
* Behavioural segmentation
* Probabilistic CLTV forecasting
* Churn prediction
* Uplift modelling for causal impact
* Customer embeddings (Customer2Vec)
* SHAP explainability
* Real-time inference via FastAPI
* Interactive analytics dashboard in Streamlit

The system is organised into modular layers to support scalability, reproducibility, and production readiness.

---

# 2. **System Architecture Diagram (Textual)**

```
 ┌───────────────────────────────────────────────────────────────────┐
 │                           Data Sources                             │
 │  (Transactional CSVs, Campaign Logs, Product Feeds, Streaming)     │
 └───────────────────────────────────────────────────────────────────┘
                  │
                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │                          Data Processing                           │
 │ - Cleaning & validation                                            │
 │ - RFM calculation (Recency, Frequency, Monetary)                   │
 │ - Tenure computation for CLTV                                      │
 │ - Feature engineering (segments, embeddings input, uplift labels)  │
 └───────────────────────────────────────────────────────────────────┘
                  │
                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │                           ML Models                                │
 │   1. Churn Model (Logistic Regression + SHAP)                      │
 │   2. Probabilistic CLTV (BG/NBD + Gamma-Gamma)                     │
 │   3. Customer2Vec Embeddings + FAISS index                         │
 │   4. Causal Uplift Model (Causal Forest / T-Learner)               │
 └───────────────────────────────────────────────────────────────────┘
                  │
                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │                     Model Serving & APIs                           │
 │ FastAPI microservice:                                              │
 │   - /predict_churn                                                 │
 │   - /predict_cltv                                                  │
 │   - /similar_customers                                             │
 │   - /uplift_score                                                  │
 │   - Utilises saved models (pkl + faiss)                            │
 └───────────────────────────────────────────────────────────────────┘
                  │
                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │                         Interactive Frontend                       │
 │                        (Streamlit Web App)                         │
 │  - Executive dashboard                                              │
 │  - Segmentation explorer                                            │
 │  - CLTV & churn analytics                                           │
 │  - SHAP explanations                                                │
 │  - Uplift simulation                                               │
 │  - Strategic campaign planner                                      │
 └───────────────────────────────────────────────────────────────────┘
                  │
                  ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │                          Data Export                               │
 │  - Segment-level exports                                            │
 │  - Targeting lists (top N customers)                                │
 │  - Model insights / SHAP reports                                    │
 └───────────────────────────────────────────────────────────────────┘
```

---

# 3. **Detailed Component Breakdown**

## **3.1 Data Layer**

### Responsibilities:

* Ingest customer transactions
* Validate schema
* Clean cancellations / negative prices
* Compute:

  * Recency
  * Frequency
  * Monetary value
  * Tenure
  * Average order value

### Key modules:

`utils/preprocessing.py`
`clean_df()`
`build_rfm()`

### Input:

Transactional CSV files (from Retail, e-commerce, CRM, ERP systems).

---

# 4. **Modelling Layer**

This is the core intelligence of the system.

---

## **4.1 Churn Prediction Model**

### Model:

`sklearn.linear_model.LogisticRegression`

### Features:

* log(Recency)
* log(Frequency)
* log(Monetary)

### Output:

`churn_prob ∈ [0, 1]`

### Explainability:

* SHAP global importance (beeswarm)
* SHAP local force or waterfall plots
* Counterfactual suggestions (Alibi)

### Stored as:

`models/churn_model.pkl` + `scaler.pkl`

---

## **4.2 Probabilistic CLTV Model**

### Frequency Model:

**BG/NBD (Fader et al.)**

### Monetary Model:

**Gamma-Gamma**

### Outputs:

* Expected purchases over horizon
* Expected average spend per purchase
* Probabilistic CLTV with uncertainty interval

### Stored as:

`models/bgf.pkl`
`models/ggf.pkl`

### Module:

`utils/cltv_models.py`

---

## **4.3 Customer Embedding Model (Customer2Vec)**

### Approach:

* Skip-gram neural model (PyTorch) or
* Matrix factorisation on co-occurrence matrix

### FAISS Index:

```
models/customer2vec.faiss
```

### Enables:

* Lookalike customer retrieval
* Similarity-based targeting
* Personalisation

### Module:

`utils/embeddings.py`

---

## **4.4 Uplift Model**

### Preferred:

**Causal Forest (Double ML)**

### Fallback:

**Two-model uplift (T-Learner)**

### Inputs:

* Treatment indicator
* Outcome variable
* Customer behaviour features

### Outputs:

`uplift_score ∈ ℝ` (incremental uplift)

### Module:

`utils/uplift.py`

---

# 5. **Model Serving Layer (FastAPI)**

The FastAPI microservice exposes model inference endpoints.

### Endpoints:

```
POST /predict_churn
POST /predict_cltv
POST /similar_customers
POST /uplift_score
```

### Responsibilities:

* Load all models into memory once
* Validate JSON payloads with Pydantic
* Serve predictions with millisecond latency

### File:

`fastapi_scorer.py`

This service supports **real-time scoring**, allowing integration with CRM systems, CDPs, or marketing automation tools.

---

# 6. **Streamlit Frontend Architecture**

The frontend is the main interface for analysts and marketers.

### Major UI Pages:

1. Executive Dashboard
2. Cluster Profiles
3. CLTV & Churn Analytics
4. AI Recommendations (campaign planner)
5. Explainability (SHAP + feature importances)
6. Uplift Simulator
7. Embedding Explorer
8. Export Tools

### Responsibilities:

* Visualise trends
* Enable interactive exploration
* Display SHAP plots
* Provide lookalike search
* Offer targeting exports
* Integrate FastAPI calls

The Streamlit layer does **no training** — only inference and visualisation.

---

# 7. **Real-Time Architecture**

If deployed in production:

```
Transactions → Kafka → Feature Store (Redis) → FastAPI Scorer → CRM/CDP
```

But for local use, you've simulated:

* A small streaming engine
* Real-time scoring dashboard
* Live churn-risk feed

---

# 8. **Model Persistence & Versioning**

All models follow semantic versioning:

```
churn_model_v1.pkl
bgf_v2.pkl
ggf_v2.pkl
customer2vec_v1.faiss
uplift_model_v1.pkl
```

Metadata stored in `/docs/model_cards/` for transparency.

---

# 9. **MLOps Considerations**

### Logging:

* API latency logs
* Feature drift checks
* CLTV calibration drift

### CI/CD:

(Optional)

* GitHub Actions to lint, test, and package the app
* Automatic dependency checks

### Deployment Targets:

* Streamlit Cloud
* Docker containers
* Azure App Service
* AWS ECS/Fargate
* GCP Cloud Run

---

# 10. **Security Architecture**

### Data protection:

* No PII is stored inside models
* CustomerID is anonymised or hashed
* Internal datasets should not be uploaded to GitHub

### API protection:

* Add authentication tokens for production
* Limit request rates
* Enable HTTPS

---

# 11. **Dependencies Overview**

Major packages:

* `lifetimes` (BG/NBD + Gamma-Gamma)
* `shap`
* `scikit-learn`
* `pandas`, `numpy`
* `plotly`
* `faiss-cpu` or `annoy`
* `fastapi`, `uvicorn`
* `torch` (optional embeddings)
* `alibi` (counterfactuals)

---

# 12. **Data Flow Diagram (Textual)**

```
Transactions CSV ─► Preprocessing ─► RFM Table ─► ML Models ─► Scores
                                                    │
                                                    ├──► FastAPI (Real-time)
                                                    │
                                                    └──► Streamlit Dashboard
```

---

# 13. **Extensibility**

The system supports modular extension:

* Swap Logistic Regression with XGBoost
* Add time-series forecasting
* Add customer journey modelling
* Integrate SQL or cloud storage
* Connect to live marketing platforms (Braze, HubSpot, Klaviyo)