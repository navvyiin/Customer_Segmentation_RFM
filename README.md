# Customer Intelligence Platform (CIP)

### An End-to-End Machine Learning Platform for Customer Analytics, Lifetime Value Prediction, Churn Intelligence, and Real-Time Decision Support

> **Customer Intelligence Platform (CIP)** is a modular machine learning system that combines customer segmentation, probabilistic lifetime value modelling, churn prediction, causal uplift modelling, representation learning, explainable AI, and real-time inference into a unified architecture for customer intelligence.

Unlike traditional customer segmentation projects that focus solely on RFM analysis or classification models, CIP integrates multiple complementary modelling paradigms into a reproducible, production-oriented workflow suitable for experimentation, research, and intelligent decision support.

---

# Motivation

Customer analytics is often implemented as a collection of disconnected models.

A marketing team may use one model for segmentation, another for churn prediction, and a third for customer lifetime value estimation, with little integration between them.

This project explores whether these analytical components can be combined into a unified intelligence platform capable of supporting data-driven customer management across the entire customer lifecycle.

Rather than optimising a single predictive model, the project focuses on designing a modular software architecture that allows multiple machine learning models to cooperate within a consistent inference pipeline.

---

# Key Capabilities

### Customer Segmentation

* Classical RFM analysis
* Behavioural segmentation
* K-Means clustering
* Interactive customer profiling

---

### Probabilistic Customer Lifetime Value

Implements

* BG/NBD
* Gamma-Gamma

to estimate

* purchase frequency
* expected future transactions
* monetary value
* customer lifetime value

---

### Churn Intelligence

Predicts customer attrition using supervised machine learning with

* feature engineering
* probability calibration
* SHAP explainability

allowing users to understand both **who is likely to churn** and **why**.

---

### Causal Uplift Modelling

Rather than predicting customer behaviour alone, the platform estimates

> **Which customers are most likely to respond positively to an intervention.**

This enables targeted marketing campaigns with reduced acquisition costs.

---

### Customer Representation Learning

Learns dense behavioural embeddings inspired by Customer2Vec and performs

* similarity search
* nearest-neighbour retrieval

using FAISS.

---

### Explainable AI

Every prediction can be interpreted through

* SHAP values
* feature importance
* local explanations

making the platform suitable for transparent decision support.

---

### Real-Time Inference

A FastAPI backend exposes prediction services for

* Churn Prediction
* CLTV
* Uplift
* Customer Similarity

allowing integration with external applications.

---

# System Architecture

```text
                 Customer Transactions
                          │
                          ▼
                 Data Preprocessing
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
     RFM Pipeline     Feature Store    Behaviour Vectors
          │               │                │
          ▼               ▼                ▼
 Segmentation       Churn Model      Customer2Vec
          │               │                │
          ├───────────────┼────────────────┤
                          ▼
                 CLTV Estimation
                          │
                          ▼
                Causal Uplift Model
                          │
                          ▼
                 Explainability Layer
                          │
                          ▼
               FastAPI Inference Engine
                          │
                          ▼
                 Streamlit Dashboard
```

---

# Machine Learning Pipeline

1. Data ingestion
2. Data validation
3. Feature engineering
4. RFM computation
5. Behavioural segmentation
6. Customer embedding generation
7. Churn prediction
8. CLTV estimation
9. Causal uplift modelling
10. Explainability
11. Real-time inference
12. Interactive visualisation

---

# Repository Structure

```text
Customer-Intelligence-Platform/

├── app.py
├── fastapi_scorer.py
├── producer_sim.py
│
├── src/
│   ├── segmentation/
│   ├── churn/
│   ├── cltv/
│   ├── uplift/
│   ├── embeddings/
│   ├── explainability/
│   ├── api/
│   └── utils/
│
├── models/
├── notebooks/
├── docs/
│
├── assets/
│   ├── architecture.png
│   ├── workflow.png
│   ├── dashboard.png
│   └── demo.gif
│
├── tests/
├── requirements.txt
└── README.md
```

---

# Technology Stack

| Layer             | Technologies          |
| ----------------- | --------------------- |
| Programming       | Python                |
| ML                | Scikit-learn, PyTorch |
| Analytics         | Pandas, NumPy         |
| CLTV              | Lifetimes             |
| Explainability    | SHAP                  |
| Similarity Search | FAISS                 |
| Backend           | FastAPI               |
| Dashboard         | Streamlit             |
| Visualisation     | Plotly, Matplotlib    |
| Deployment        | Render                |

---

# Engineering Challenges

Building individual machine learning models is relatively straightforward.

The primary challenge of this project was designing a software architecture capable of integrating heterogeneous modelling paradigms into a reusable inference pipeline.

Each analytical component—customer segmentation, probabilistic lifetime value estimation, churn prediction, causal uplift modelling, and behavioural embeddings—operates under different assumptions and requires distinct preprocessing pipelines.

A modular architecture was therefore developed to minimise code duplication, isolate model-specific logic, and provide consistent interfaces for both interactive exploration and API-based inference.

This project demonstrates that production-oriented machine learning is fundamentally a software engineering problem as much as a modelling problem.

---

# Current Limitations

* Batch inference only
* Single-node execution
* No distributed training
* No feature store
* No experiment tracking
* No model versioning
* Limited automated testing

---

# Future Directions

* MLflow experiment tracking
* Feast feature store
* Kubeflow pipelines
* Docker containers
* Kubernetes deployment
* Kafka streaming
* Airflow orchestration
* Real-time monitoring
* Drift detection
* Model registry
* A/B testing framework
* Vector databases for customer retrieval
* LLM-powered campaign generation

---

# Reproducibility

Every experiment can be reproduced using the provided datasets, configuration files, and documented execution pipeline. Random seeds are fixed where appropriate to improve reproducibility.

---

# Documentation

```
docs/

architecture.md

system_design.md

methodology.md

model_card.md

limitations.md

future_work.md

benchmark.md

api.md
```

---

# Assets

```
assets/

demo.gif

dashboard.png

architecture.png

workflow.png

model_pipeline.png
```

---

# Citation

```text
If you use Customer Intelligence Platform in academic work, please cite:

Naval Kishore
Customer Intelligence Platform: An End-to-End Machine Learning System for Customer Analytics and Decision Support.
GitHub Repository, 2026.
```

---

# License

MIT License
