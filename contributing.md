# 🤝 Contributing Guide

**Customer Segmentation & Intelligence Platform**
*Advanced RFM • Probabilistic CLTV • Causal Uplift • Explainable AI • Real-Time Scoring*

---

## Welcome 👋

Thank you for your interest in contributing to the **Customer Segmentation & Intelligence Platform**.

This project brings together **advanced data science, machine learning, causal inference, customer lifetime value (CLTV) modelling, neural embeddings, explainable AI, and real-time MLOps engineering**.

Our goal is to maintain a **high-quality, ethical, and collaborative environment** where contributors of all experience levels can meaningfully participate.

Please read this guide carefully **before opening an issue or pull request**.

---

## 📁 Project Overview

```
Customer_Segmentation_RFM/
│
├── app.py                       # Streamlit dashboard
├── fastapi_scorer.py            # Real-time inference service
├── requirements.txt
├── README.md
├── .gitignore
│
├── utils/
│   ├── cltv_models.py           # BG/NBD + Gamma-Gamma models
│   ├── uplift.py                # Uplift & causal inference
│   ├── embeddings.py            # Customer2Vec + FAISS
│   ├── shap_utils.py            # SHAP explainability
│   └── preprocessing.py         # Data cleaning & transforms
│
├── models/                      # Serialized models (avoid large binaries)
├── data/                        # Sample or synthetic datasets only
└── docs/                        # Architecture diagrams & model cards
```

---

## ✅ Prerequisites

Before contributing, ensure you have:

### Required

* **Python 3.10**

  > Other versions are unsupported due to ML dependency constraints.

### Recommended Environment Setup

```bash
python -m venv rfm_env
source rfm_env/bin/activate      # macOS / Linux
rfm_env\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Optional (Advanced Modelling)

Depending on your contribution, you may also need:

* PyTorch (embeddings)
* faiss-cpu (similarity search)
* shap
* alibi (counterfactuals)
* econml or causalml (OS/Python dependent)

---

## 🔁 How to Contribute

All contributions must follow this workflow:

### 1. Fork the Repository

Click **Fork** on GitHub.

### 2. Create a Feature Branch

Use the following naming convention:

```bash
git checkout -b feature/<area>-<short-description>
```

**Examples**

* `feature/shap-improvements`
* `feature/probabilistic-cltv`
* `feature/customer2vec-embeddings`

---

### 3. Make Your Changes

* Keep code **modular**, **readable**, and **documented**
* Follow the coding standards below

---

### 4. Format, Lint, and Type-Check

Before committing, run:

```bash
black .
flake8 .
mypy .
```

---

### 5. Commit Using Conventional Commits

**Examples**

```
feat: add SHAP decision plot for churn model
fix: correct BG/NBD convergence fallback
perf: optimise embedding similarity search
docs: update architecture diagram
```

---

### 6. Push and Open a Pull Request

```bash
git push origin feature/<your-branch>
```

Your PR should include:

* ✔ What you changed
* ✔ Why the change matters
* ✔ Screenshots (for Streamlit UI changes)
* ✔ Benchmarks (for ML changes)
* ✔ Risks, assumptions, or limitations

A maintainer will review your PR as soon as possible.

---

## 🧑‍💻 Coding Guidelines

### Python Standards

* Follow **PEP 8**
* Use **type hints everywhere**
* Add **docstrings** to public functions
* Keep functions **small, testable, and modular**
* Avoid deeply nested logic

---

### 🤖 Machine Learning Standards

When modifying ML components:

* Always set:

  ```python
  random_state = 42
  ```
* Clearly document:

  * Model assumptions
  * Hyperparameters
  * Training logic
* Use `joblib.dump()` for model persistence
* Avoid pickling with mismatched library versions

If retraining models, include:

* ROC / PR curves
* SHAP feature importance
* Drift checks where applicable

---

### 🖥 Streamlit UI Guidelines

* Keep heavy computation **outside UI logic**
* Use caching appropriately:

  ```python
  st.cache_resource  # models
  st.cache_data      # processed datasets
  ```
* Avoid blocking operations (e.g. training models in UI)
* Prefer **Plotly** over Matplotlib for interactive visuals

---

### ⚡ FastAPI Guidelines

* Use **Pydantic** schemas for requests/responses
* Ensure responses are **JSON-serialisable**
* Load ML models **once** using global cache
* Avoid heavy computation inside request handlers

---

## 🧪 Testing Guidelines

Even minimal testing is encouraged. Use **pytest**.

### Recommended Tests

**Preprocessing**

* Missing columns
* Incorrect dtypes
* Invalid or negative values

**CLTV Models**

* BG/NBD fit stability
* Gamma-Gamma convergence or fallback

**Uplift Models**

* Numeric outputs from T-learner / causal forest

**Embeddings**

* FAISS index returns valid neighbours

**FastAPI**

```python
client = TestClient(app)
response = client.post("/predict", json=payload)
```

**Streamlit**

* Unit test utility functions only

---

## 🌿 Branching Model

We follow a simple and stable branching strategy:

* `main` — production-ready, versioned releases
* `dev` — integration branch for upcoming release
* `feature/*` — individual contributions

🚫 Direct commits to `main` are not allowed.

---

## 📋 Pull Request Requirements

Every PR must include:

* Clear description of changes
* Screenshots (if UI-related)
* Benchmarks (if ML-related)
* Risk assessment (new dependencies, breaking changes)
* Documentation updates where relevant

---

## 🔐 Security & Ethical ML Guidelines

This project involves sensitive analytics workflows.

* ❌ Do not commit real or proprietary customer data
* ✔ Use anonymised or synthetic datasets only
* Avoid embedding PII in models or embeddings
* Review SHAP outputs for fairness or bias concerns
* Document known limitations and ethical considerations

---

## 🧾 Model Versioning Policy

Models in `/models` must follow this naming scheme:

```
churn_model_v1.pkl
bgf_v2.pkl
customer2vec_v1.faiss
```

For each update:

* Increment the version
* Add or update a **model card** in `/docs/model_cards/`
* Update changelog
* Remove obsolete models unless required

---

## 🆘 Requesting Support

If you encounter issues:

1. Search existing GitHub issues
2. Open a new issue including:

   * Full error stack trace
   * OS and Python version
   * Steps to reproduce
   * Screenshots (if UI-related)

Maintainers will respond as soon as possible.

---

## 🙏 Thank You

Your contributions help evolve this **enterprise-grade customer intelligence platform**.

We appreciate your time, expertise, and commitment —
**let’s build something impactful together.** 🚀
