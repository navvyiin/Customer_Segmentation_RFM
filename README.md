# 🧠 Customer Segmentation & Intelligence Platform

An **end-to-end customer analytics and intelligence platform** built with Python, combining **RFM segmentation**, **CLTV forecasting**, **churn prediction**, **uplift modeling**, **customer embeddings**, **explainable AI**, and **real-time inference**.

This project goes beyond traditional RFM analysis by delivering a **production-style machine learning system** with dashboards, APIs, and modular pipelines suitable for real-world marketing, retention, and personalization use cases.

---

## 🚀 What This Project Does

This platform enables teams to:

* Segment customers based on behavioral patterns
* Forecast customer lifetime value probabilistically
* Predict churn risk and explain predictions
* Measure marketing impact using causal uplift models
* Identify similar customers using learned embeddings
* Serve real-time predictions via an API
* Explore insights interactively through a web UI

---

## ✨ Core Features

### 🔍 Customer Segmentation

* RFM (Recency, Frequency, Monetary) scoring
* Automated segment labeling
* Visual diagnostics and summaries

### 📈 Forecasting & Risk

* Probabilistic **CLTV modeling** (BG/NBD + Gamma-Gamma)
* **Churn prediction** with feature-level explainability (SHAP)

### 🎯 Personalization & Marketing

* **Uplift modeling** to estimate treatment effects
* Campaign prioritization and targeting logic

### 🧠 Embeddings & Similarity

* **Customer2Vec-style embeddings**
* Fast similarity search using FAISS

### 🛠 Interfaces & Tooling

* Interactive **Streamlit dashboard**
* **FastAPI** real-time inference service
* Event simulation for live scoring pipelines
* Modular, reusable ML components

---

## 🧩 Architecture Overview

This repository is structured like a **production ML system**, not a notebook demo.

```
Customer_Segmentation_RFM/
├── app.py                     # Streamlit dashboard
├── fastapi_scorer.py          # FastAPI inference service
├── producer_sim.py            # Real-time event simulator
├── rfm_utils.py               # RFM computation utilities
├── cltv_models.py             # CLTV models
├── uplift.py                  # Causal uplift modeling
├── embeddings.py              # Customer embeddings & FAISS
├── explainability.py          # SHAP explainability
├── campaign_planner.py        # Campaign decision logic
├── models/                    # Saved model artifacts
├── docs/
│   ├── architecture.md        # System design
│   └── model_card.md          # Model documentation
├── requirements.txt
├── contributing.md
└── code_of_conduct.md
```

---

## 📦 Installation

### Prerequisites

* Python **3.8+**
* Virtual environment recommended

### Setup

```bash
git clone https://github.com/navvyiin/Customer_Segmentation_RFM.git
cd Customer_Segmentation_RFM
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### 📊 Streamlit Dashboard

```bash
streamlit run app.py
```

Use the dashboard to:

* Explore RFM segments
* View CLTV and churn predictions
* Inspect explainability plots
* Analyze customer similarity

---

### ⚡ Real-Time Inference API

```bash
uvicorn fastapi_scorer:app --reload --port 8001
```

Example endpoints:

* `POST /predict_churn`
* `POST /predict_cltv`
* `POST /uplift_score`
* `POST /similar_customers`

---

### 🌀 Event Simulation (Optional)

```bash
python producer_sim.py
```

Simulates streaming customer events for real-time scoring pipelines.

---

## 🧠 Use Cases

This platform is suitable for:

* 📦 Customer segmentation & profiling
* 📉 Churn prediction and retention strategy
* 💰 Customer lifetime value forecasting
* 🎯 Campaign targeting & uplift analysis
* 🤝 Recommendation and similarity search
* 🧪 ML system design demonstrations

---

## 🛠 Technologies Used

* **Python**
* **Pandas / NumPy**
* **Scikit-learn**
* **Lifetimes**
* **SHAP**
* **FAISS**
* **FastAPI**
* **Streamlit**

---

## ⚠️ Notes & Limitations

* Designed for experimentation, prototyping, and learning
* Performance depends on dataset size and hardware
* Not intended as a plug-and-play SaaS product

---

## 🛣 Roadmap

* Dockerized deployment
* Model monitoring & drift detection
* Feature store integration
* CI/CD and automated testing
* Cloud-native deployment examples

---

## 🤝 Contributing

Contributions, ideas, and improvements are welcome.
Please review `contributing.md` before submitting changes.

---

## 📄 License

MIT License
© 2026 navvyiin
