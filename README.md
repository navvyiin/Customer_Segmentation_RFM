# Customer Intelligence Platform

## Production-Oriented Machine Learning for Customer Analytics, Retention, and Personalisation

> A modular customer intelligence platform that integrates customer segmentation, probabilistic lifetime value estimation, churn prediction, causal uplift modelling, representation learning, explainable AI, and real-time inference into a single deployable system.

---

## Motivation

Customer analytics is often fragmented.

Traditional customer segmentation focuses on descriptive analysis, while predictive modelling, customer lifetime value estimation, recommendation systems, explainability, and campaign optimisation are usually implemented as separate workflows.

This project was developed to demonstrate how modern customer intelligence systems can integrate these capabilities into a unified machine learning architecture suitable for experimentation, deployment, and future production environments.

---

### Customer Behaviour Analysis

* RFM feature engineering
* Behavioural segmentation
* K-Means clustering

---

### Customer Lifetime Value

* BG/NBD
* Gamma-Gamma
* Probabilistic CLTV estimation

---

### Customer Retention

* Churn prediction
* Feature importance
* SHAP explanations

---

### Marketing Optimisation

* Uplift modelling
* Treatment effect estimation
* Campaign targeting

---

### Representation Learning

* Customer2Vec embeddings
* FAISS similarity search
* Nearest neighbour retrieval

---

### Real-Time ML

* FastAPI inference
* Live scoring
* Streaming simulation

---

# Architecture

```text
Customer Data
      │
      ▼
Feature Engineering
      │
      ▼
────────────────────────────────────────────
│          Behaviour Modelling             │
│                                          │
│  RFM        CLTV        Churn            │
│  KMeans     BG/NBD      XGBoost          │
│                                          │
────────────────────────────────────────────
      │
      ▼
Representation Learning
(Customer2Vec + FAISS)
      │
      ▼
Campaign Optimisation
(Causal Uplift)
      │
      ▼
Explainability (SHAP)
      │
      ▼
FastAPI Inference Layer
      │
      ▼
Streamlit Dashboard
```

---

# Why this project is different

This platform combines descriptive analytics, probabilistic modelling, predictive machine learning, causal inference, representation learning, explainable AI, and deployment into a unified architecture, illustrating how customer intelligence systems are designed in modern production environments.

---

## Engineering Challenges

The most difficult aspect of this project was not implementing individual machine learning models, but designing a modular architecture capable of integrating heterogeneous modelling paradigms.

Each analytical component—including probabilistic CLTV estimation, supervised churn prediction, causal uplift modelling, and embedding-based similarity search—requires different preprocessing assumptions, feature pipelines, and inference workflows.

Developing reusable interfaces while avoiding duplicated logic required careful separation between data engineering, feature engineering, model orchestration, and API layers.

---

# Repository Structure

```
Customer-Intelligence-Platform/

app/

models/

pipelines/

api/

experiments/

docs/

assets/

tests/

docker/

configs/

```
---

# Machine Learning Pipeline

```
Raw Customer Data
       │
       ▼
Cleaning
       │
       ▼
Feature Engineering
       │
       ▼
─────────────────────────────
│ RFM                     │
│ CLTV                    │
│ Churn                   │
│ Embeddings              │
│ Uplift                  │
─────────────────────────────
       │
       ▼
Model Registry
       │
       ▼
FastAPI
       │
       ▼
Dashboard
```

---

# Design Principles

* Modular architecture
* Separation of concerns
* Reproducibility
* Explainability
* Deployment-first design
* Extensibility

---

# Limitations

Current limitations include

* single-node execution

* no distributed training

* no feature store

* no experiment tracking

* no automated retraining

* no model monitoring

* no Kubernetes deployment

---

# Future Work

* MLflow experiment tracking

* Kubeflow pipelines

* Feature store integration

* Online learning

* Reinforcement learning for campaign optimisation

* LLM-powered customer insights

* Vector databases

* Streaming inference using Kafka

* Distributed deployment with Kubernetes

* Model monitoring with Evidently AI

---
