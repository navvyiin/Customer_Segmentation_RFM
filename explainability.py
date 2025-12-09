import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


def fit_surrogate_ltv_model(rfm: pd.DataFrame):
    """
    Train a RandomForest surrogate model to approximate LTV
    using Recency, Frequency, Monetary as features.
    """
    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features].values
    y = rfm["LTV"].values

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model, X, features


def get_shap_explainer(model, X_background, nsamples: int = 200):
    """
    Build a SHAP explainer on a sample of the background data.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is not installed.")

    if X_background.shape[0] > nsamples:
        background = shap.sample(X_background, nsamples, random_state=42)
    else:
        background = X_background

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(background)
    return explainer, shap_vals


def simple_counterfactual_churn(
    churn_model,
    scaler,
    cust_row: pd.Series,
    target_prob: float = 0.25,
) -> str:
    """
    Very simple local counterfactual:
    - Uses log-transformed RFM like the churn model.
    - Tries to see if reducing Recency and modestly lifting Frequency / Monetary
      could push churn risk below target_prob.
    """
    r = float(cust_row["Recency"])
    f = float(cust_row["Frequency"])
    m = float(cust_row["Monetary"])

    base = np.array([[np.log1p(r), np.log1p(f), np.log1p(m)]])
    base_scaled = scaler.transform(base)
    p0 = float(churn_model.predict_proba(base_scaled)[:, 1])

    if p0 <= target_prob:
        return (
            f"Current estimated churn risk is {p0:.1%}, "
            f"already below the target threshold of {target_prob:.0%}."
        )

    for delta_days in [30, 60, 90]:
        r_new = max(r - delta_days, 1)
        f_new = f + 1
        m_new = m * 1.05

        trial = np.array([[np.log1p(r_new), np.log1p(f_new), np.log1p(m_new)]])
        trial_scaled = scaler.transform(trial)
        p_new = float(churn_model.predict_proba(trial_scaled)[:, 1])

        if p_new <= target_prob:
            return (
                f"Model suggests churn risk could drop from {p0:.1%} to {p_new:.1%} "
                f"if you can re-engage this customer within ~{delta_days} days, "
                "nudge one extra purchase and slightly increase spend per order."
            )

    return (
        f"Estimated churn risk is {p0:.1%}. The model does not find a small, "
        f"local change to Recency/Frequency/Monetary that brings risk below {target_prob:.0%}. "
        "A stronger intervention or deeper incentive may be needed."
    )