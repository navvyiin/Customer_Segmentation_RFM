import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import ConvergenceError

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter

    LIFETIMES_AVAILABLE = True
except ImportError:
    BetaGeoFitter = None
    GammaGammaFitter = None
    LIFETIMES_AVAILABLE = False


def prepare_lifetimes_summary(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    """
    Prepare summary data in the form required by lifetimes:
        customer_id, frequency, recency, T, monetary_value
    """
    d = df.copy()
    d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"])

    g = (
        d.groupby("CustomerID")
        .agg(
            first_purchase=("InvoiceDate", "min"),
            last_purchase=("InvoiceDate", "max"),
            monetary=("TotalPrice", "sum"),
            n_invoices=("InvoiceNo", "nunique"),
        )
        .reset_index()
    )

    g["frequency"] = (g["n_invoices"] - 1).clip(lower=0)
    g["recency"] = (g["last_purchase"] - g["first_purchase"]).dt.days
    g["T"] = (ref_date - g["first_purchase"]).dt.days
    g["monetary_value"] = g["monetary"] / g["n_invoices"].replace(0, np.nan)

    g = g[g["monetary_value"].notna()].copy()

    summary = g[
        [
            "CustomerID",
            "frequency",
            "recency",
            "T",
            "monetary_value",
        ]
    ].reset_index(drop=True)
    return summary


def fit_bg_g_and_gg(summary, penalizer_coef=0.1):
    """
    Fit BG/NBD and Gamma-Gamma models robustly with automatic fallbacks.
    """

    # --------------------------
    # Clean and prepare summary
    # --------------------------
    df = summary.copy()

    # Remove customers with frequency zero or invalid spend
    df = df[(df["frequency"] > 0) & (df["monetary_value"] > 0)]

    # Remove extreme outliers (top 0.5%)
    df = df[df["monetary_value"] < df["monetary_value"].quantile(0.995)]

    # Remove customers with only 1 purchase (GG cannot estimate heterogeneity well)
    df_gg = df[df["frequency"] > 1]

    # --------------------------
    # Fit BG/NBD model
    # --------------------------
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
    bgf.fit(df["frequency"], df["recency"], df["T"])

    # --------------------------
    # Fit Gamma-Gamma robustly
    # --------------------------
    ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)

    try:
        ggf.fit(df_gg["frequency"], df_gg["monetary_value"])
    except ConvergenceError:
        # Retry with stronger penalisation
        try:
            ggf = GammaGammaFitter(penalizer_coef=penalizer_coef * 10)
            ggf.fit(df_gg["frequency"], df_gg["monetary_value"])
        except ConvergenceError:
            # Final fallback
            print("Gamma-Gamma failed to converge. Using deterministic AOV model.")
            return bgf, None  # signal deterministic fallback

    return bgf, ggf


def predict_cltv(bgf, ggf, summary, months=12):
    df = summary.copy()

    # Predict purchase counts
    df["pred_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        months,
        df["frequency"],
        df["recency"],
        df["T"]
    )

    # If Gamma-Gamma failed → deterministic spend model
    if ggf is None:
        df["pred_cltv"] = df["pred_purchases"] * df["monetary_value"].mean()
        df["model_used"] = "BG/NBD + deterministic spend"
        return df

    # Otherwise use full GG
    df["exp_avg_value"] = ggf.conditional_expected_average_profit(
        df["frequency"], df["monetary_value"]
    )

    df["pred_cltv"] = df["pred_purchases"] * df["exp_avg_value"]
    df["model_used"] = "BG/NBD + Gamma-Gamma"

    return df