import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning for transactional retail data."""
    df = df.copy()
    df = df[df["CustomerID"].notna()]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_rfm(df: pd.DataFrame):
    """Build Recency, Frequency, Monetary table from cleaned transactions."""
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )

    rfm = rfm[rfm["Monetary"] > 0].reset_index(drop=True)
    return rfm, ref_date


def scale_rfm(rfm: pd.DataFrame):
    """Log-transform and standardise RFM for clustering / modelling."""
    rfm_log = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_log)
    return scaled, scaler


def compute_k_options(scaled: np.ndarray, k_min: int, k_max: int):
    """Compute KMeans inertia and silhouette for a range of K."""
    inertias, silhouettes = [], []
    K_values = list(range(k_min, k_max + 1))

    for k in K_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        inertias.append(km.inertia_)
        try:
            silhouettes.append(silhouette_score(scaled, labels))
        except Exception:
            silhouettes.append(float("nan"))

    return K_values, inertias, silhouettes


def auto_name_segments(rfm: pd.DataFrame):
    """Assign human-readable names to clusters based on z-scored R/F/M."""
    summary = (
        rfm.groupby("Cluster")
        .agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
        )
    )

    z = (summary - summary.mean()) / summary.std(ddof=0)
    labels = {}

    for cid, row in z.iterrows():
        r, f, m = row["Recency"], row["Frequency"], row["Monetary"]

        if r < -0.6 and f > 0.8 and m > 0.8:
            labels[cid] = "Premium Loyalists"
        elif f > 0.5 and m > 0.3:
            labels[cid] = "Active Repeat Buyers"
        elif m > 0.9 and r > 0.6:
            labels[cid] = "High-Value Dormant"
        elif r > 0.7 and f < -0.4:
            labels[cid] = "At-Risk / Churn Likely"
        elif m < -0.6 and f < -0.6:
            labels[cid] = "Bargain-Driven Low Value"
        else:
            labels[cid] = "Emerging Customers"

    return labels


def churn_model_and_scores(
    rfm_scaled: np.ndarray,
    rfm: pd.DataFrame,
    recency_threshold_days: int = 180,
):
    """
    Simple churn model: label = 1 if Recency > threshold else 0.
    Logistic regression on scaled log-RFM.
    """
    rfm = rfm.copy()
    y = (rfm["Recency"] > recency_threshold_days).astype(int)
    X = rfm_scaled

    model = LogisticRegression(max_iter=500)

    try:
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        # fallback: scale Recency to [0,1] as crude proxy
        probs = np.clip(rfm["Recency"] / rfm["Recency"].max(), 0, 1)

    rfm["churn_prob"] = probs
    return model, rfm


def compute_ltv_deterministic(
    rfm: pd.DataFrame,
    df_period_years: float,
    profit_margin: float = 0.3,
    horizon_years: int = 3,
) -> pd.DataFrame:
    """
    Deterministic, heuristic LTV:
        LTV ≈ AOV * annual_freq * margin * expected_years
    where expected_years is a simple function of (1 - churn_prob).
    """
    rfm = rfm.copy()

    rfm["avg_order_value"] = rfm["Monetary"] / rfm["Frequency"].replace(0, np.nan)
    rfm["annual_freq"] = rfm["Frequency"] / max(df_period_years, 1 / 365)

    rfm["avg_order_value"].fillna(rfm["Monetary"], inplace=True)
    rfm["expected_years"] = (1 - rfm["churn_prob"]) * horizon_years
    rfm["expected_years"] = rfm["expected_years"].clip(
        lower=0.25, upper=horizon_years
    )

    rfm["LTV"] = (
        rfm["avg_order_value"]
        * rfm["annual_freq"]
        * profit_margin
        * rfm["expected_years"]
    )

    return rfm