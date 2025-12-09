# **Corporate RFM Analytics Suite**

### *An End-to-End Customer Intelligence Platform with Segmentation, Probabilistic CLTV, Churn Prediction, Causal Uplift, Customer2Vec Embeddings, SHAP Explainability, and Real-Time Scoring*

---

<div align="center">

**Built with Python, Streamlit, FastAPI, scikit-learn, lifetimes, SHAP, FAISS, PyTorch, and modern MLOps practices**
**Designed for enterprise analytics, retention science, and personalised marketing at scale**

</div>

---

# **Why This Project Matters**

This platform goes far beyond simple RFM segmentation.
It demonstrates the *full lifecycle* of a modern ML system:

* Data ingestion
* Statistical modelling
* Machine learning
* Causal analysis
* Embedding learning
* Real-time inference
* Interpretability
* API serving
* Interactive dashboards

Most portfolio projects show only notebooks or static models —
**this one shows a full production-ready architecture**.

---

# **Key Features**

### **1. Advanced Behavioural Segmentation (RFM + K-Means)**

* Automated RFM computation
* Dynamic cluster search with Silhouette & Elbow diagnostics
* Auto-naming of segments using statistical heuristics
* Behavioural radar plots and revenue attribution

---

### **2. Probabilistic CLTV (BG/NBD + Gamma-Gamma)**

* Industry-standard lifetime value forecasting
* Per-customer probability distributions, not point estimates
* Cohort-based uncertainty bands
* Convergence-safe training with penalisation
* Clear model interpretation

---

### **3. Churn Risk Estimation with SHAP Explainability**

* Logistic regression baseline (production-friendly)
* SHAP global and local explanations
* Decision plots & force plots
* Counterfactual suggestions (“minimal behaviour change needed to avoid churn”)

---

### **4. Customer2Vec Embeddings**

* Learn dense vector representations of customers
* Capture behavioural similarity beyond RFM
* Fast FAISS similarity search
* Lookalike modelling for targeted acquisition
* Interactive similarity explorer

---

### **5. Causal Uplift Modelling**

* Estimate the *incremental* effect of a marketing treatment
* Causal Forest (double ML) with fallback T-Learner
* Uplift curves, Qini, AUUC
* Identify who should *receive* offers and who would buy anyway
* Targeting based on profitability, not conversion.

---

### **6. AI-Driven Campaign Planner**

* Segment-level uplift
* Prioritised campaign ranking
* Budget allocation engine
* Personalised message templates
* Exportable targeting lists

---

### **7. Real-Time ML Serving (FastAPI)**

Endpoints include:

```
POST /predict_churn
POST /predict_cltv
POST /similar_customers
POST /uplift_score
```

Fully compatible with:

* Webhooks
* CRM systems
* Marketing automation platforms
* Real-time scoring pipelines

---

### **8. Streaming Simulation**

A Kafka-style simulator producing events into FastAPI for live scoring:

* Near real-time churn monitoring
* Live “at-risk customer feed” in Streamlit
* Event-driven ML architecture demonstration

---

### **9. Responsible AI**

* SHAP explanations
* Counterfactual reasoning
* Model cards
* Architecture documentation
* Clear limitations and ethical use guidelines

---

# **Architecture Overview**

```
📁 Customer_Segmentation_RFM
│
├── app.py                  # Streamlit interface (main UI)
├── fastapi_scorer.py       # Real-time inference service
├── producer_sim.py         # Streaming simulation engine
│
├── utils/
│   ├── rfm_utils.py        # Cleaning, RFM computation
│   ├── cltv_models.py      # BG/NBD + Gamma-Gamma modelling
│   ├── uplift.py           # Causal uplift modelling
│   ├── embeddings.py       # Customer2Vec + FAISS
│   ├── explainability.py   # SHAP + counterfactuals
│   ├── realtime.py         # Stream helper utilities
│   ├── campaign_planner.py # AI-driven marketing recommendations
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   ├── (future) bgf.pkl, ggf.pkl, uplift.pkl, customer2vec.faiss
│
├── data/
│   └── sample.csv (safe example dataset)
│
├── docs/
│   ├── architecture.md
│   ├── code_of_conduct.md
│   ├── contributing.md
│   ├── model_card.md
│
└── requirements.txt
```

This structure mirrors real production ML systems.

---

# **How to Run the Project**

## **1. Clone the Repository**

```bash
git clone https://github.com/<your-username>/Customer_Segmentation_RFM.git
cd Customer_Segmentation_RFM
```

## **2. Create Virtual Environment (Python 3.10 Recommended)**

```bash
python -m venv rfm_env
.\rfm_env\Scripts\activate
```

## **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

## **4. Run Streamlit Dashboard**

```bash
streamlit run app.py
```

## **5. Run FastAPI Microservice**

```bash
uvicorn fastapi_scorer:app --reload --port 8001
```

## **6. Optional: Start Streaming Simulator**

```bash
python producer_sim.py
```

---

# **Screenshots / Demo**

*Add screenshots or GIFs here of your dashboard, SHAP plots, embeddings explorer, campaign planner, etc.*
(If you want, I can generate a text layout for screenshots.)

---

# **Tech Stack**

| Category       | Tools                                            |
| -------------- | ------------------------------------------------ |
| UI             | Streamlit                                        |
| API            | FastAPI + Uvicorn                                |
| ML             | scikit-learn, lifetimes, PyTorch                 |
| Embeddings     | FAISS                                            |
| Explainability | SHAP, Alibi                                      |
| Causal ML      | EconML / CausalML / fallback T-Learner           |
| Data           | Pandas, NumPy                                    |
| Visualisation  | Plotly                                           |
| MLOps          | Model persistence, versioning, real-time scoring |

---

# **What This Project Demonstrates to Recruiters**

### ✔ Ability to design and build large-scale ML systems

### ✔ Strong mathematical grounding (probabilistic models, causal inference)

### ✔ Real-time model serving with FastAPI

### ✔ Production-style modular code organization

### ✔ Advanced interpretability and Responsible AI tools

### ✔ Enterprise-grade architecture and documentation

### ✔ A polished, interactive analytics application

This project is the kind that gets shortlisted.

---

# **Contact**

For questions or collaboration opportunities, feel free to reach out:

**Naval Kishore**
**usnavalg@gmail.com**
**https://www.linkedin.com/in/navalkishore2005/**