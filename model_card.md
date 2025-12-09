# **MODEL_CARD.md**

# **Corporate RFM Analytics Suite — Model Card**

This Model Card describes the machine learning models included in the Corporate RFM Analytics Suite. It provides information about intended use, data requirements, assumptions, known limitations, risks, and ethical considerations.

---

# **1. Model Overview**

This application contains multiple production-grade analytical and machine learning models:

| Model                            | Purpose                                                | Method                           |
| -------------------------------- | ------------------------------------------------------ | -------------------------------- |
| **Churn Prediction Model**       | Predict customer churn risk                            | Logistic Regression              |
| **CLTV Model (Probabilistic)**   | Forecast Customer Lifetime Value                       | BG/NBD + Gamma-Gamma             |
| **Customer2Vec Embedding Model** | Learn dense vector representations for customers       | Skip-gram / Matrix Factorisation |
| **Uplift Model**                 | Estimate incremental impact of marketing interventions | Causal Forest or T-Learner       |

Each model supports a different layer of customer intelligence.

---

# **2. Intended Use**

These models are intended for:

### ✔ Customer segmentation

### ✔ Lifetime value forecasting

### ✔ Churn prevention & retention strategies

### ✔ Personalised targeting & lookalike discovery

### ✔ Causal ROI estimation for marketing campaigns

### ✔ Strategic planning & predictive analytics

They are **not intended for:**

* Credit scoring
* Employment decisions
* Healthcare prioritisation
* Any high-stakes decisions involving individuals

---

# **3. Training Data**

### Source

The models are trained on **transaction-level customer purchase history**, including:

* CustomerID
* InvoiceNo
* Invoice Date
* Quantity
* Unit price

### Preprocessing steps

* Remove cancelled orders
* Filter non-positive quantities or prices
* Create RFM metrics: Recency, Frequency, Monetary
* Compute tenure (T) for CLTV models
* Aggregate product interactions for embeddings
* Derive binary treatment/outcome for uplift modelling (if available)

### Important

No PII (personal identifiable information) is stored inside the models.

---

# **4. Model Details**

---

## **4.1 Churn Prediction Model**

**Type:** Logistic Regression
**Features:**

* Recency
* Frequency
* Monetary value
* Log-transformed / standardised

**Output:** Probability of churn (0–1).

**Strengths:**

* Easy to interpret
* Fast, stable, requires little data
* Works well with SHAP explainability

**Limitations:**

* Linear relationships only
* Cannot capture complex behavioural patterns

---

## **4.2 Probabilistic CLTV Model**

**Models:**

* **BG/NBD** → predicts frequency of future transactions
* **Gamma-Gamma** → predicts monetary value per transaction

**Outputs:**

* Expected number of purchases over horizon
* Expected monetary value
* Full LTV distribution (probabilistic)
* Confidence intervals

**Strengths:**

* Industry standard for subscription/e-commerce
* Probabilistic uncertainty is explicit
* Works well with sparse data

**Limitations:**

* Assumes purchase independence
* Requires frequency ≥ 1 for Gamma-Gamma
* Can fail to converge (fallback included)

---

## **4.3 Customer2Vec Embedding Model**

**Type:** Skip-gram / Co-occurrence / Autoencoder (depending on configuration)
**Input:** Product sequences or co-purchase graphs
**Output:** 32–128 dimensional customer vectors
**Uses:**

* Lookalike targeting
* Similar customer retrieval via FAISS
* Personalisation

**Strengths:**

* Captures behavioural nuance beyond RFM
* Scales well
* Enables nearest-neighbour marketing strategies

**Limitations:**

* Requires sufficient product diversity
* Embeddings may encode hidden biases
* Sensitive to window size and training frequency

---

## **4.4 Uplift Model**

**Type:**

* Preferred: Causal Forest (Double ML)
* Fallback: T-learner (two-model uplift)

**Output:** Estimated incremental lift per customer (uplift score).

**Strengths:**

* Targets customers who react positively to treatments
* Avoids waste (discounts to customers who would buy anyway)
* Supports ROI-based campaign optimisation

**Limitations:**

* Requires randomised or quasi-random treatment assignment
* Can be unstable with small sample sizes
* Sensitive to confounding if treatment logs are poor

---

# **5. Evaluation**

### Churn Model (Logistic Regression)

| Metric             | Value              |
| ------------------ | ------------------ |
| AUC                | Depends on dataset |
| Precision/Recall   | Dataset-dependent  |
| Feature importance | Available via SHAP |

### CLTV (BG/NBD + Gamma-Gamma)

* RMSE of predicted vs actual spend
* Calibration plots (expected vs observed frequency)
* Distribution-fitted log-likelihood

### Customer2Vec

* Intrinsic evaluation: cosine similarity coherence
* Extrinsic evaluation: downstream uplift in targeting

### Uplift Model

* Qini curve
* Uplift@K
* AUUC (Area Under Uplift Curve)

---

# **6. Risks & Limitations**

### Statistical Risks

* BG/NBD may underperform with irregular purchase cycles
* Gamma-Gamma fails for extreme monetary variance
* Logistic model may oversimplify churn dynamics

### Operational Risks

* Embeddings require retraining periodically
* Uplift depends on treatment logging quality
* Real-time scoring must avoid stale models

### Ethical Risks

* Embeddings may cluster customers in ways that unintentionally reflect socioeconomic patterns
* Models may over-prioritise high-value customers

Mitigation steps:

* Use SHAP to audit feature impact
* Review segment-level outcomes
* Avoid discriminatory targeting

---

# **7. Privacy Considerations**

* Do not train on raw personal data
* CustomerID is hashed or anonymised internally
* No demographic attributes are used
* Embeddings do not store raw identifiers

---

# **8. Model Retraining Policy**

| Model        | Retraining Frequency                       |
| ------------ | ------------------------------------------ |
| Churn Model  | Every 3 months or after behavioural shifts |
| CLTV Model   | Monthly                                    |
| Customer2Vec | Weekly or after catalogue changes          |
| Uplift Model | After each major campaign cycle            |

---

# **9. Model Versioning**

All models follow semantic versioning:

```
churn_model_v1.pkl
bgf_v2.pkl
customer2vec_v1.faiss
uplift_model_v3.pkl
```

Model Card updates accompany each new version.

---

# **10. Responsible Use**

Users of this model should ensure:

* Predictions are not used for high-stakes individual decisions
* Campaigns are fair, privacy-preserving, and transparent
* Outputs are monitored for drift and bias
* Business actions based on model scores are human-reviewed

---

# **11. Contact**

For questions or concerns about these models, contact:

**[usnavalg@gmail.com](mailto:usnavalg@gmail.com)**