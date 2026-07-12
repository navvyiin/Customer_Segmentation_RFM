# Corporate RFM Analytics Suite

An end-to-end customer intelligence platform combining RFM segmentation, probabilistic lifetime value modelling, churn prediction, and uplift analysis.

**Live app:** https://customer-segmentation-rfm-usng.streamlit.app/

## What it does

Processes 397,884 transaction records down to 4,338 unique customers (£8.91M total revenue), then:

- **Segments customers via K-Means clustering** on Recency, Frequency, and Monetary features into four groups: Premium Loyalists (705 customers, £5.75M revenue, 12.4% avg churn), At-Risk/Churn Likely (1,617 customers, £0.57M revenue, 99.6% churn probability), Emerging Customers, and Bargain-Driven Low Value.
- **Models customer lifetime value probabilistically** using the BG/NBD model (via the `lifetimes` library), producing an average CLTV of £1,704.95 rather than a single deterministic point estimate — this matters because CLTV is inherently a distribution, not a fixed number, and treating it as one hides how uncertain the estimate is for any individual customer.
- **Predicts churn** with an overall churn probability of 19.8% across the customer base.
- **Explains predictions** using SHAP values, so segment assignment and churn risk aren't black-box outputs.
- **Serves inference in real time** through a FastAPI backend, structured into 7 modules (RFM utilities, CLTV modelling, uplift analysis, embeddings, explainability, campaign planning, real-time serving).

## Why BG/NBD instead of a simpler CLTV heuristic

A common shortcut is estimating CLTV as `average order value × purchase frequency × customer lifespan`, which assumes every customer's future behaviour matches their historical average — it doesn't account for the fact that a customer who hasn't purchased in six months is probably not still active. BG/NBD models purchase timing and dropout as separate stochastic processes, which is why it can distinguish a customer who's likely churned from one who's just an infrequent purchaser, something a flat average can't do.

## Tech stack

Python, Streamlit, FastAPI, Scikit-learn, `lifetimes`, SHAP, FAISS, PyTorch, Plotly.

## Known limitations

- Segmentation and CLTV modelling were run on a single retail transaction dataset; the segment boundaries and BG/NBD parameters would need re-fitting for a different customer base or industry.
- Uplift analysis is implemented but wasn't validated against a genuine randomized holdout — the causal claims from it should be treated as exploratory rather than confirmed.

## Setup

```bash
git clone https://github.com/navvyiin/corporate-rfm-analytics-suite.git
cd corporate-rfm-analytics-suite
pip install -r requirements.txt
uvicorn api.main:app --reload &
streamlit run dashboard/app.py
```
