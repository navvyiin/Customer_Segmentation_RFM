**Corporate RFM Analytics Suite**
Advanced Segmentation • Probabilistic CLTV • Causal Uplift • SHAP Explainability • Real-Time Scoring

---

# Welcome

Thank you for your interest in contributing to the **Corporate RFM Analytics Suite**.
This project combines advanced data science, machine learning, causal inference, CLTV modelling, neural embeddings, and real-time MLOps engineering.

The goal of this document is to create a consistent, safe, and collaborative environment for contributors of all skill levels.

Please read this guide before opening a Pull Request.

---

# Project Overview

```
Customer_Segmentation_RFM/
│
├── app.py                       # Streamlit interface
├── fastapi_scorer.py            # Real-time inference service
├── requirements.txt
├── README.md
├── .gitignore
│
├── utils/
│   ├── cltv_models.py           # BG/NBD + Gamma-Gamma modelling
│   ├── uplift.py                # Uplift modelling + Causal inference
│   ├── embeddings.py            # Customer2Vec embeddings + FAISS search
│   ├── shap_utils.py            # SHAP explainability + decision plots
│   └── preprocessing.py         # Cleaning + transformations
│
├── models/                      # Serialized models, avoid committing binaries
├── data/                        # Sample datasets only
└── docs/                        # Architecture diagrams, model cards
```

---

# Prerequisites

Before contributing, ensure you have:

### ✔ Python 3.10

(Other versions are unsupported due to ML dependencies.)

### ✔ Recommended setup

```bash
python -m venv rfm_env
.\rfm_env\Scripts\activate
pip install -r requirements.txt
```

### ✔ Additional optional packages for modelling:

* PyTorch (for embeddings)
* faiss-cpu (for similarity search)
* alibi (counterfactuals)
* shap
* econml / causalml (if supported by your OS/Python version)

---

# How to Contribute

Follow this workflow for all contributions:

---

## Fork the repository

Click “Fork” on GitHub.

---

## Create a feature branch

Use the following naming scheme:

```
git checkout -b feature/<area>-<short-description>
```

Examples:

```
feature/shap-improvements
feature/probabilistic-cltv
feature/customer2vec-embedding
```

---

## Make your changes

Keep code modular and documented.
Follow the guidelines in the next sections.

---

## Format, lint, and type-check code

Before committing:

```bash
black .
flake8 .
mypy .
```

---

## Commit using *conventional commits*

Examples:

```
feat: add SHAP decision plot for churn model
fix: correct BG/NBD convergence fallback
perf: optimise embedding similarity search
docs: update architecture diagram
```

---

## Push and open a Pull Request

```
git push origin feature/<your-branch>
```

In your PR, include:

* What you changed
* Why the change matters
* Screenshots (for Streamlit UI changes)
* Performance benchmarks if modifying models
* Any risks or assumptions

A reviewer will engage with you shortly.

---

# Coding Guidelines

High-quality code is important for this project.
Please follow these principles:

---

## Python Style

* Follow **PEP8**
* Use **type hints** everywhere
* Add **docstrings** to public functions
* Keep functions small, modular, and testable
* Avoid deeply nested logic if possible

---

## Machine Learning Style

When modifying ML components:

### Always set random_state=42

Ensures reproducibility.

### Include a description of the model

Document hyperparameters in comments.

### Do not pickle with mismatched sklearn versions

Prefer `joblib.dump` + lock version in `requirements.txt`.

### Include model validation

If retraining, provide:

* ROC / PR curves
* SHAP feature importances
* Drift checks if applicable

---

## Streamlit UI Guidelines

* Keep complex computation outside the UI (use caching)
* Use:

```
st.cache_resource  # for models
st.cache_data      # for processed datasets
```

* Avoid blocking operations (e.g., training models inside UI)
* Prefer Plotly over Matplotlib for interactive plots

---

## FastAPI Guidelines

* Use Pydantic for request/response schemas
* Ensure API responses are JSON-serialisable
* Avoid heavy ML in the request loop
  → Load models once using global cache

---

# Testing Guidelines

Even if tests are limited, use `pytest`.

Recommended tests:

### Preprocessing

Missing columns, incorrect dtypes, negative prices.

### CLTV modelling

* BG/NBD fit should not crash
* Gamma-Gamma model should converge or trigger fallback

### Causal uplift modelling

Ensure T-learner or causal forest produces numeric outputs.

### Embeddings

FAISS index should return neighbours.

### FastAPI endpoints

Test with `httpx`:

```python
client = TestClient(app)
response = client.post("/predict", json=payload)
```

### Streamlit functions

Use unit tests for utility functions only.

---

# Branching Model

We follow a simple, stable branching model:

### `main`

Production-ready code, versioned releases.

### `dev`

Integration branch for next release.

### `feature/*`

Individual feature branches.

**No one commits directly to `main`.**
All changes go through Pull Requests.

---

# Pull Request Requirements

Every PR must include:

* Clear explanation of the change
* Screenshots (if UI elements changed)
* Benchmark results for ML model updates
* Risk assessment (e.g., new dependencies)
* Updates to docs if needed (e.g., new features)

---

# Security & Ethical ML Guidelines

This project involves sensitive customer data modelling.
To ensure safety:

### Do not commit production datasets

Only use anonymised samples.

### Avoid embedding PII in models

Do not store raw identifiers in embeddings.

### Respect model fairness

Check SHAP explanations for unintended bias signals.

### Document known issues

If a model has blind spots, describe them.

---

# Model Versioning Policy

Models in `/models` should follow:

```
churn_model_v1.pkl
bgf_v2.pkl
customer2vec_v1.faiss
```

For each update:

* Increment version
* Add a model card to `/docs/model_cards/`
* Update changelog
* Remove obsolete versions unless needed

---

# Requesting Support

If you encounter issues:

1. Search existing GitHub issues
2. Open a new issue with:

   * Full error stack trace
   * OS / Python version
   * Steps to reproduce
   * Screenshots if UI-related

The maintainers will respond ASAP.

---

# Thank You

Your contributions help improve this enterprise-grade RFM analytics platform.
We appreciate your time and expertise — let’s build something remarkable together.
