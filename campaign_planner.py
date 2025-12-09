import numpy as np
import pandas as pd


def recommendations_for_segment(name: str, stats: dict):
    ltv = stats.get("LTV_mean", stats.get("avg_LTV"))
    churn = stats.get("churn_prob_mean", stats.get("avg_churn"))

    recs = []

    if ltv is None or churn is None:
        return ["Insufficient data to generate recommendations."]

    if ltv >= 500 and churn < 0.25:
        recs.append("Protect loyalty — VIP experiences, exclusive early access.")
        recs.append("Upsell premium bundles and subscription upgrades.")

    if ltv >= 300 and churn >= 0.25:
        recs.append("Reactivation campaign tailored to past behaviour.")
        recs.append("Investigate friction points (service, product fit).")

    if ltv < 300 and churn < 0.4:
        recs.append("Nurturing flows — education, cross-sell campaigns.")
        recs.append("Encourage repeat habits (reminders, bundles).")

    if churn >= 0.5:
        recs.append("Emergency retention flow — highly personalised offers.")
        recs.append("Pause paid acquisition; focus on retention.")

    if not recs:
        recs.append("General lifecycle improvements recommended.")

    return recs


def assign_campaign(seg: pd.DataFrame):
    """
    Take a Segment-level dataframe with columns:
        Segment, avg_LTV, avg_churn, customers, revenue
    and enrich it with recommended_campaign, priority_score, expected uplift, etc.
    """
    seg = seg.copy()

    total_revenue = seg["revenue"].sum()
    seg["revenue_share"] = seg["revenue"] / total_revenue

    campaign_templates = {
        "Protect & Reward": {
            "description": "High-LTV, low-churn customers. Protect and deepen relationship.",
            "channels": ["Email", "SMS", "Loyalty App"],
            "suggested_offer": "Exclusive bundle, early access, VIP service",
            "expected_uplift": 0.06,
            "budget_pct": 0.04,
        },
        "Win-Back Personalised": {
            "description": "High-LTV but rising churn — reactive personalised outreach.",
            "channels": ["Email", "Phone Outreach", "Direct Mail"],
            "suggested_offer": "Personalised discount, tailored picks",
            "expected_uplift": 0.12,
            "budget_pct": 0.10,
        },
        "Growth Nurture": {
            "description": "Mid-LTV, low churn — increase frequency and basket size.",
            "channels": ["Email", "In-App", "Paid Social"],
            "suggested_offer": "Cross-sell flows, content-led upsell",
            "expected_uplift": 0.08,
            "budget_pct": 0.06,
        },
        "Cost-Efficient Conversion": {
            "description": "Low LTV, price sensitive. Efficient activation.",
            "channels": ["Performance Ads", "Referral", "SMS"],
            "suggested_offer": "Time-limited discount, referral rewards",
            "expected_uplift": 0.04,
            "budget_pct": 0.03,
        },
        "Emergency Retention": {
            "description": "Very high churn probability — aggressive retention needed.",
            "channels": ["Phone Outreach", "Email", "SMS"],
            "suggested_offer": "High-value personalised offer, loyalty win-back",
            "expected_uplift": 0.20,
            "budget_pct": 0.18,
        },
    }

    def pick_campaign(row):
        if row["avg_LTV"] > seg["avg_LTV"].quantile(0.75) and row["avg_churn"] < 0.25:
            return "Protect & Reward"
        if row["avg_LTV"] > seg["avg_LTV"].quantile(0.6) and row["avg_churn"] >= 0.25:
            return "Win-Back Personalised"
        if row["avg_LTV"] >= seg["avg_LTV"].quantile(0.4) and row["avg_churn"] < 0.35:
            return "Growth Nurture"
        if row["avg_churn"] >= 0.5:
            return "Emergency Retention"
        return "Cost-Efficient Conversion"

    seg["recommended_campaign"] = seg.apply(pick_campaign, axis=1)

    seg["priority_score"] = (
        seg["avg_LTV"]
        * (1 - seg["avg_churn"])
        * np.sqrt(seg["customers"])
        * seg["revenue_share"]
    )

    seg["expected_uplift"] = seg["recommended_campaign"].apply(
        lambda x: campaign_templates[x]["expected_uplift"]
    )
    seg["est_incremental_revenue"] = seg["revenue"] * seg["expected_uplift"]
    seg["est_budget"] = (
        seg["est_incremental_revenue"] * seg["revenue_share"] * 0.6
        + seg["revenue"] * seg["expected_uplift"] * 0.02
    )

    return seg, campaign_templates