import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import shap

try:
    from alibi.explainers import AnchorTabular # type: ignore
    ALIBI_AVAILABLE = True
except ImportError:
    ALIBI_AVAILABLE = False
from utils.rfm_utils import (
    clean_df,
    build_rfm,
    scale_rfm,
    compute_k_options,
    auto_name_segments,
    churn_model_and_scores,
    compute_ltv_deterministic,
)
from utils.cltv_models import (
    prepare_lifetimes_summary,
    fit_bg_g_and_gg,
    predict_cltv,
    LIFETIMES_AVAILABLE,
)
from utils.explainability import (
    fit_surrogate_ltv_model,
    get_shap_explainer,
    simple_counterfactual_churn,
    SHAP_AVAILABLE,
)
from utils.embeddings import (
    build_customer_product_matrix,
    compute_svd_embeddings,
    build_similarity_index,
    query_similarity,
    FAISS_AVAILABLE,
    ANNOY_AVAILABLE,
)
from utils.uplift import (
    prepare_uplift_data,
    train_drl_uplift_model,
    predict_uplift,
    uplift_table,
)
from utils.realtime import FASTAPI_SCORER, PRODUCER_SIM
from utils.campaign_planner import (
    recommendations_for_segment,
    assign_campaign,
)

# ----------------- PAGE CONFIG & THEME -----------------
st.set_page_config(
    page_title="Corporate RFM Suite — Extended",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#0b6e6b"
BG = "#f7faf9"
CARD = "#ffffff"
TEXT = "#0b2b2a"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {BG}; color: {TEXT}; }}
    .big-font {{ font-size:20px; font-weight:600; color:{PRIMARY_COLOR}; }}
    .kpi {{ background:{CARD}; padding:14px; border-radius:8px;
           box-shadow: 0 4px 18px rgba(0,0,0,0.04); }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
[data-testid="stMetricValue"] {
    color: #0b2b2a !important;
}
[data-testid="stMetricDelta"] {
    color: #0b6e6b !important;
}
[data-testid="stMetricLabel"] {
    color: #0b2b2a !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------- DATA LOADING -----------------
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
campaign_uploaded = st.sidebar.file_uploader(
    "Optional: Upload campaign log (CSV) for uplift", type=["csv"]
)

LOCAL_FALLBACK = "data.csv"


def read_csv_flexible(file_obj):
    if isinstance(file_obj, str):
        return pd.read_csv(file_obj, encoding="ISO-8859-1")
    file_obj.seek(0)
    return pd.read_csv(file_obj, encoding="ISO-8859-1")


if uploaded is not None:
    df_raw = read_csv_flexible(uploaded)
else:
    if os.path.exists(LOCAL_FALLBACK):
        df_raw = read_csv_flexible(LOCAL_FALLBACK)
        st.sidebar.info(f"Using fallback: {LOCAL_FALLBACK}")
    else:
        st.sidebar.error("Upload a CSV or place data.csv next to app.py")
        st.stop()

required = {"CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"}
missing = required - set(df_raw.columns)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df = clean_df(df_raw)
st.success(f"Loaded dataset: {len(df):,} rows")

# ----------------- SIDEBAR SETTINGS -----------------
profit_margin = st.sidebar.slider("Profit Margin", 0.05, 0.7, 0.30, 0.05)
horizon_years = st.sidebar.slider("LTV Horizon (years)", 1, 5, 3)
recency_threshold = st.sidebar.slider(
    "Churn Recency Threshold (days)", 90, 365, 180
)
k_min, k_max = st.sidebar.slider("Cluster Search Range", 2, 12, (2, 10))
initial_k = st.sidebar.slider("Selected K", k_min, k_max, 4)
compute_shap = st.sidebar.checkbox(
    "Compute SHAP explanations (can be slower)", value=False
)
run_embeddings_flag = st.sidebar.checkbox(
    "Build customer embeddings (SVD + Faiss/Annoy/bruteforce)", value=True
)

# ----------------- CORE RFM + CLUSTERING -----------------
rfm, ref_date = build_rfm(df)

days_period = (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days
years_period = max(days_period / 365, 1 / 365)

scaled, scaler = scale_rfm(rfm)

K_vals, inertias, silhs = compute_k_options(scaled, k_min, k_max)
selected_k = initial_k

kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=20)
rfm["Cluster"] = kmeans.fit_predict(scaled)

segment_map = auto_name_segments(rfm)
rfm["Segment"] = rfm["Cluster"].map(segment_map)

churn_model, rfm = churn_model_and_scores(
    scaled, rfm, recency_threshold_days=recency_threshold
)

rfm = compute_ltv_deterministic(
    rfm, years_period, profit_margin, horizon_years
)

cluster_stats = (
    rfm.groupby("Cluster")
    .agg(
        customers=("CustomerID", "count"),
        Recency_mean=("Recency", "mean"),
        Frequency_mean=("Frequency", "mean"),
        Monetary_mean=("Monetary", "mean"),
        churn_prob_mean=("churn_prob", "mean"),
        LTV_mean=("LTV", "mean"),
        Revenue_sum=("Monetary", "sum"),
    )
    .reset_index()
)
cluster_stats["Segment"] = cluster_stats["Cluster"].map(segment_map)

# ----------------- NAVIGATION -----------------
pages = [
    "Executive Overview",
    "Cluster Profiles",
    "LTV & Churn",
    "CLTV (Probabilistic)",
    "Explainability",
    "Embeddings",
    "Uplift Modelling",
    "Realtime (example)",
    "AI Recommendations",
    "Export",
]
page = st.sidebar.radio("Navigate", pages)

# ----------------- EXECUTIVE OVERVIEW -----------------
if page == "Executive Overview":
    st.markdown(
        "<div class='big-font'>Executive Dashboard</div>",
        unsafe_allow_html=True,
    )

    total_customers = rfm["CustomerID"].nunique()
    total_revenue = rfm["Monetary"].sum()
    avg_ltv = rfm["LTV"].mean()
    avg_churn = rfm["churn_prob"].mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        f"<div class='kpi'><b>Total Customers</b><br><h3>{total_customers:,}</h3></div>",
        unsafe_allow_html=True,
    )
    k2.markdown(
        f"<div class='kpi'><b>Total Revenue (£)</b><br><h3>{total_revenue:,.0f}</h3></div>",
        unsafe_allow_html=True,
    )
    k3.markdown(
        f"<div class='kpi'><b>Average LTV (£)</b><br><h3>{avg_ltv:.2f}</h3></div>",
        unsafe_allow_html=True,
    )
    k4.markdown(
        f"<div class='kpi'><b>Avg Churn Risk</b><br><h3>{avg_churn:.1%}</h3></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### K Selection Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.line(x=K_vals, y=inertias, markers=True, title="Elbow Curve"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.line(x=K_vals, y=silhs, markers=True, title="Silhouette Scores"),
            use_container_width=True,
        )

    st.markdown("### Cluster Distribution")
    dist = rfm["Segment"].value_counts().reset_index()
    dist.columns = ["Segment", "Customers"]
    st.plotly_chart(
        px.bar(dist, x="Segment", y="Customers", color="Segment"),
        use_container_width=True,
    )

    st.markdown("### Revenue by Segment")
    rev = (
        cluster_stats.groupby("Segment")["Revenue_sum"]
        .sum()
        .reset_index()
    )
    st.plotly_chart(
        px.pie(rev, names="Segment", values="Revenue_sum"),
        use_container_width=True,
    )

# ----------------- CLUSTER PROFILES -----------------
elif page == "Cluster Profiles":
    st.markdown(
        "<div class='big-font'>Cluster Profiles</div>",
        unsafe_allow_html=True,
    )

    seg_choice = st.selectbox(
        "Select Segment", sorted(rfm["Segment"].unique())
    )
    cid = [k for k, v in segment_map.items() if v == seg_choice][0]
    seg_df = rfm[rfm["Cluster"] == cid]

    st.markdown(
        f"### {seg_choice} — {len(seg_df):,} customers, Revenue £{seg_df['Monetary'].sum():,.2f}"
    )

    radar = seg_df[["Recency", "Frequency", "Monetary"]].copy()
    norm = (
        lambda x: (x - x.min()) / (x.max() - x.min())
        if x.max() != x.min()
        else x * 0
    )

    r = 1 - norm(radar["Recency"])
    f = norm(radar["Frequency"])
    m = norm(radar["Monetary"])
    vals = [r.mean(), f.mean(), m.mean()]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=vals + [vals[0]],
            theta=["Recency↑", "Frequency", "Monetary", "Recency↑"],
            fill="toself",
            name=seg_choice,
        )
    )
    fig.update_layout(
        title="Behaviour Radar", polar=dict(radialaxis=dict(range=[0, 1]))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Recency", f"{seg_df['Recency'].mean():.1f}")
    c2.metric("Avg Frequency", f"{seg_df['Frequency'].mean():.2f}")
    c3.metric("Avg Monetary (£)", f"{seg_df['Monetary'].mean():.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Avg LTV (£)", f"{seg_df['LTV'].mean():.2f}")
    c5.metric("Avg Churn Risk", f"{seg_df['churn_prob'].mean():.1%}")
    c6.metric("Total Revenue (£)", f"{seg_df['Monetary'].sum():.2f}")

    st.markdown("### Recommended Actions")
    recs = recommendations_for_segment(
        seg_choice,
        {
            "LTV_mean": seg_df["LTV"].mean(),
            "churn_prob_mean": seg_df["churn_prob"].mean(),
        },
    )
    for r in recs:
        st.success(r)

# ----------------- LTV & CHURN (simple distributions) -----------------
elif page == "LTV & Churn":
    st.markdown(
        "<div class='big-font'>LTV Forecasting & Churn Analytics</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Churn Probability Distribution")
    st.plotly_chart(
        px.histogram(rfm, x="churn_prob", nbins=30),
        use_container_width=True,
    )

    st.subheader("Customer LTV Distribution")
    st.plotly_chart(
        px.histogram(rfm, x="LTV", nbins=30),
        use_container_width=True,
    )

    st.subheader("Segment LTV vs Churn Risk")
    seg = (
        rfm.groupby("Segment")
        .agg(
            LTV=("LTV", "mean"),
            churn_prob=("churn_prob", "mean"),
            CustomerID=("CustomerID", "count"),
        )
        .reset_index()
    )
    fig = px.scatter(
        seg,
        x="LTV",
        y="churn_prob",
        size="CustomerID",
        color="Segment",
        title="Segments: LTV vs Churn",
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------- CLTV (Probabilistic) -----------------
elif page == "CLTV (Probabilistic)":
    st.markdown(
        "<div class='big-font'>Probabilistic CLTV — BG/NBD + Gamma-Gamma</div>",
        unsafe_allow_html=True,
    )

    if not LIFETIMES_AVAILABLE:
        st.warning("Install `lifetimes` to enable this page.")
    else:
        summary = prepare_lifetimes_summary(df, ref_date)
        st.write(f"Prepared lifetimes summary for {len(summary):,} customers.")
        st.dataframe(summary.head())

        with st.spinner("Fitting BG/NBD and Gamma-Gamma models..."):
            bgf, ggf = fit_bg_g_and_gg(summary)
        st.success("Models fitted.")

        months = st.slider("Prediction horizon (months)", 1, 36, 12)
        summary_pred = predict_cltv(bgf, ggf, summary, months=months)

        st.markdown("### Predicted CLTV distribution")
        st.plotly_chart(
            px.histogram(
                summary_pred,
                x="pred_cltv",
                nbins=40,
                title=f"Predicted CLTV over {months} months",
            ),
            use_container_width=True,
        )

        summary_pred = summary_pred.merge(
            rfm[["CustomerID", "Segment"]],
            on="CustomerID",
            how="left",
        )
        seg_cltv = (
            summary_pred.groupby("Segment")
            .agg(
                avg_cltv=("pred_cltv", "mean"),
                median_cltv=("pred_cltv", "median"),
                customers=("CustomerID", "count"),
            )
            .reset_index()
        )
        st.markdown("### Segment-level CLTV")
        st.dataframe(
            seg_cltv.style.format(
                {"avg_cltv": "{:.2f}", "median_cltv": "{:.2f}"}
            )
        )

        st.download_button(
            "Download CLTV predictions (CSV)",
            summary_pred.to_csv(index=False).encode("utf-8"),
            "cltv_predictions.csv",
            "text/csv",
        )

# ----------------- EXPLAINABILITY -----------------
elif page == "Explainability":
    st.markdown(
        "<div class='big-font'>Model Explainability & Responsible AI</div>",
        unsafe_allow_html=True,
    )

    st.write(
        "This section opens up the churn model: you can see what drives predictions globally, "
        "zoom into a single customer, and explore rule-based explanations and counterfactual-style suggestions."
    )

    # Features used by churn model (log-transformed & scaled)
    feature_cols = ["Recency", "Frequency", "Monetary"]
    X = rfm[feature_cols].copy()
    X_log = np.log1p(X)
    X_scaled = scaler.transform(X_log)

    # -----------------------------
    # SHAP Explainer (cached)
    # -----------------------------
    @st.cache_resource
    def load_shap_explainer():
        # use closure over churn_model and X_scaled
        explainer = shap.Explainer(churn_model, X_scaled)
        return explainer

    try:
        explainer = load_shap_explainer()
        shap_values = explainer(X_scaled)
    except Exception as e:
        st.error(f"SHAP could not be computed: {e}")
        st.stop()

    # -----------------------------
    # Alibi AnchorTabular (optional)
    # -----------------------------
    if ALIBI_AVAILABLE:
        @st.cache_resource
        def load_anchor_explainer():
            # prediction function: takes raw (unscaled) X, applies same preprocessing
            def predict_fn(x):
                x_df = pd.DataFrame(x, columns=feature_cols)
                x_log = np.log1p(x_df)
                x_scaled_local = scaler.transform(x_log)
                proba = churn_model.predict_proba(x_scaled_local)[:, 1]
                return (proba > 0.5).astype(int)

            expl = AnchorTabular(predict_fn, feature_names=feature_cols)
            expl.fit(X.values)
            return expl

        try:
            anchor_explainer = load_anchor_explainer()
        except Exception as e:
            st.warning(f"Alibi AnchorTabular initialisation failed: {e}")
            anchor_explainer = None
    else:
        anchor_explainer = None

    # -----------------------------
    # Tabs layout
    # -----------------------------
    tab_global, tab_local, tab_decision, tab_cf = st.tabs(
        [
            "🌍 Global importance",
            "🧍 Local (per-customer)",
            "🧮 Decision plot",
            "🔧 Counterfactuals & rules",
        ]
    )

    # ===========================================
    # TAB 1 — GLOBAL FEATURE IMPORTANCE
    # ===========================================
    with tab_global:
        st.subheader("Global feature importance")

        shap.initjs()

        # --- Plotly bar chart: mean |SHAP| importance ---
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        fi_df = pd.DataFrame(
            {"feature": feature_cols, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False)

        st.markdown("#### Mean |SHAP| importance (Plotly)")
        fig_bar = px.bar(
            fi_df,
            x="feature",
            y="mean_abs_shap",
            title="Mean absolute SHAP value by feature",
        )
        fig_bar.update_layout(xaxis_title="Feature", yaxis_title="Mean |SHAP|")
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- SHAP summary plots (Matplotlib) ---
        st.markdown("#### SHAP summary plots")

        # Bar-type summary
        fig_shap_bar, ax1 = plt.subplots(figsize=(8, 4))
        shap.summary_plot(
            shap_values.values, X, plot_type="bar", show=False
        )
        plt.tight_layout()
        st.pyplot(fig_shap_bar)

        # Beeswarm summary
        fig_shap_bw, ax2 = plt.subplots(figsize=(8, 5))
        shap.summary_plot(shap_values.values, X, show=False)
        plt.tight_layout()
        st.pyplot(fig_shap_bw)

        st.caption(
            "Interpretation: larger |SHAP| = stronger average influence on churn prediction. "
            "Colour in the beeswarm shows whether high/low feature values drive risk up or down."
        )

    # ===========================================
    # TAB 2 — LOCAL EXPLANATION (SINGLE CUSTOMER)
    # ===========================================
    with tab_local:
        st.subheader("Explain a single customer's churn score")

        cust_id = st.selectbox(
            "Select Customer", rfm["CustomerID"].sort_values().unique()
        )
        idx = rfm.index[rfm["CustomerID"] == cust_id][0]
        row = rfm.loc[idx]

        st.metric("Churn probability", f"{row['churn_prob']:.2%}")

        st.markdown("#### SHAP waterfall plot")
        st.write(
            "Shows how each feature pushes this customer’s churn risk up or down relative to the average."
        )

        try:
            fig_wf, ax = plt.subplots(figsize=(9, 5))
            shap.plots.waterfall(shap_values[idx], show=False)
            plt.tight_layout()
            st.pyplot(fig_wf)
        except Exception as e:
            st.warning(f"Waterfall plot not available: {e}")

        st.markdown("#### Raw RFM profile")
        st.json(
            {
                "CustomerID": str(cust_id),
                "Recency": float(row["Recency"]),
                "Frequency": float(row["Frequency"]),
                "Monetary": float(row["Monetary"]),
                "Approx. LTV": float(row["LTV"]),
                "Churn probability": float(row["churn_prob"]),
                "Segment": str(row["Segment"]),
            }
        )

    # ===========================================
    # TAB 3 — SHAP DECISION PLOT
    # ===========================================
    with tab_decision:
        st.subheader("SHAP decision plot (trajectory of decisions)")

        st.write(
            "This visualises how the model’s baseline churn risk gets adjusted by features across multiple customers."
        )

        # Choose a slice of customers to keep the plot readable
        n_samples = st.slider(
            "Number of customers to display in the decision plot",
            min_value=10,
            max_value=min(200, len(rfm)),
            value=50,
            step=10,
        )

        # Take top-n by churn risk or random sample
        mode = st.radio(
            "Sample mode",
            ["Top by churn risk", "Random sample"],
            index=0,
            horizontal=True,
        )

        if mode == "Top by churn risk":
            idxs = rfm.sort_values("churn_prob", ascending=False).head(
                n_samples
            ).index
        else:
            idxs = np.random.choice(
                rfm.index, size=n_samples, replace=False
            )

        try:
            fig_dec, ax = plt.subplots(figsize=(10, 6))
            shap.plots.decision(shap_values[idxs], show=False)
            plt.tight_layout()
            st.pyplot(fig_dec)
        except Exception as e:
            st.warning(f"Decision plot not available with this SHAP backend: {e}")
            st.write(
                "If you're using the new SHAP Explanation objects, some plot types may not be implemented yet."
            )

    # ===========================================
    # TAB 4 — COUNTERFACTUALS & ALIBI RULES
    # ===========================================
    with tab_cf:
        st.subheader("Counterfactual-style suggestions & Alibi rules")

        st.write(
            "Here we combine two views:\n"
            "- a simple numeric counterfactual search (what behaviour change lowers churn?), and\n"
            "- Alibi’s AnchorTabular rules (if installed), which give human-readable conditions that lock in the prediction."
        )

        cust_id_cf = st.selectbox(
            "Select customer for counterfactual analysis",
            rfm["CustomerID"].sort_values().unique(),
            key="cf_cust_select",
        )
        idx_cf = rfm.index[rfm["CustomerID"] == cust_id_cf][0]
        row_cf = rfm.loc[idx_cf]

        current_prob = float(row_cf["churn_prob"])
        st.metric("Current churn probability", f"{current_prob:.2%}")

        # -------- Heuristic numeric counterfactual search --------
        st.markdown("#### Heuristic numeric counterfactual")

        def search_counterfactual(rec, freq, mon, target_prob_drop=0.10):
            """
            Very small grid search over changes in Recency and Frequency.
            Returns the best local change that reduces predicted churn.
            """
            best = None
            best_prob = current_prob

            for d_rec in range(0, 61, 5):  # reduce recency by up to 60 days
                for d_freq in range(0, 6):  # add up to 5 purchases
                    rec_new = max(rec - d_rec, 0)
                    freq_new = freq + d_freq
                    mon_new = mon  # keep constant for simplicity

                    X_test = pd.DataFrame(
                        [[rec_new, freq_new, mon_new]],
                        columns=feature_cols,
                    )
                    X_test_log = np.log1p(X_test)
                    X_test_scaled = scaler.transform(X_test_log)
                    p_new = churn_model.predict_proba(X_test_scaled)[0, 1]

                    if p_new < best_prob:
                        best_prob = p_new
                        best = (d_rec, d_freq, p_new)

            return best, best_prob

        best_change, new_prob = search_counterfactual(
            row_cf["Recency"], row_cf["Frequency"], row_cf["Monetary"]
        )

        if best_change is not None:
            d_rec, d_freq, p_new = best_change
            st.success(
                f"- Reduce effective recency by **{d_rec} days** "
                f"(e.g. triggered re-engagement within that window)\n"
                f"- Increase purchase count by **{d_freq}** over the horizon\n\n"
                f"Estimated churn reduction: **{current_prob - p_new:.2%}** "
                f"(down to ~{p_new:.2%})."
            )
        else:
            st.info(
                "No strong local improvement found within the small search grid. "
                "This customer may need a more radical intervention."
            )

        # -------- Alibi AnchorTabular rules --------
        st.markdown("#### Alibi AnchorTabular local rules")

        if anchor_explainer is None:
            if not ALIBI_AVAILABLE:
                st.info(
                    "Install `alibi` to enable rule-based explanations:\n"
                    "`pip install alibi`"
                )
            else:
                st.warning("AnchorTabular explainer is not available (initialisation failed).")
        else:
            instance = X.loc[[idx_cf]].values
            try:
                explanation = anchor_explainer.explain(instance[0])
                st.write("**Anchor rule:**")
                st.code(" AND ".join(explanation.anchor))
                st.write(
                    f"Precision: {explanation.precision:.2f}, "
                    f"Coverage: {explanation.coverage:.2f}"
                )
                st.caption(
                    "Interpretation: if the listed conditions hold, the model’s prediction "
                    "is very likely to stay the same. These are rule-style, human-readable "
                    "local explanations around the chosen customer."
                )
            except Exception as e:
                st.warning(f"Failed to compute Alibi anchor explanation: {e}")

# ----------------- EMBEDDINGS -----------------
elif page == "Embeddings":
    st.markdown(
        "<div class='big-font'>Customer Embeddings & Similarity</div>",
        unsafe_allow_html=True,
    )

    if not run_embeddings_flag:
        st.info(
            "Enable 'Build customer embeddings' in the sidebar to compute embeddings."
        )
    else:
        try:
            with st.spinner("Building customer × product matrix..."):
                cust_prod = build_customer_product_matrix(df)
            st.write(
                f"Matrix shape: {cust_prod.shape[0]} customers × {cust_prod.shape[1]} products"
            )

            n_dims = st.slider("Embedding dimensionality", 8, 256, 64, 8)
            embeddings, cust_ids, _ = compute_svd_embeddings(
                cust_prod, n_components=n_dims
            )
            st.success("Embeddings computed.")

            index, kind = build_similarity_index(embeddings)
            st.write(
                f"Similarity index: {kind} (Faiss: {FAISS_AVAILABLE}, Annoy: {ANNOY_AVAILABLE})"
            )

            q_customer = st.selectbox("Query customer", cust_ids)
            q_idx = cust_ids.index(q_customer)
            q_vec = embeddings[q_idx]
            top_k = st.slider("Top-K similar customers", 3, 50, 10)

            idxs, dists = query_similarity(
                index, embeddings, q_vec, top_k=top_k, kind=kind
            )

            st.markdown("### Nearest neighbours")
            for rank, (i, dist) in enumerate(zip(idxs, dists), 1):
                st.write(f"{rank}. {cust_ids[i]} — distance: {dist:.3f}")

            if st.button("Download embeddings (CSV)"):
                emb_df = pd.DataFrame(embeddings, index=cust_ids)
                emb_df.reset_index(inplace=True)
                emb_df.rename(columns={"index": "CustomerID"}, inplace=True)
                st.download_button(
                    "Click to download",
                    emb_df.to_csv(index=False).encode("utf-8"),
                    "customer_embeddings.csv",
                    "text/csv",
                )
        except Exception as e:
            st.error(f"Embeddings failed: {e}")

# ----------------- UPLIFT MODELLING (DR Learner only) -----------------
elif page == "Uplift Modelling":
    st.markdown(
        "<div class='big-font'>Uplift / Incremental Impact Modelling</div>",
        unsafe_allow_html=True,
    )

    st.write(
        "This module estimates which customers generate *incremental* revenue when targeted, "
        "not just who is likely to convert."
    )

    if campaign_uploaded is not None:
        st.info("Using uploaded campaign log for uplift modelling.")
        try:
            campaign_df = read_csv_flexible(campaign_uploaded)
            st.dataframe(campaign_df.head())
        except Exception as e:
            st.error(f"Failed to read campaign log: {e}")
            campaign_df = None
    else:
        st.info(
            "No campaign log uploaded — generating a synthetic campaign from the current RFM cohort."
        )
        rng = np.random.default_rng(42)
        campaign_df = rfm.copy()
        n = len(campaign_df)

        # Random treatment assignment (50%)
        campaign_df["treatment"] = rng.binomial(1, 0.5, size=n)

        # Synthetic outcome linked to Monetary, LTV, and treatment
        base = 0.3 * campaign_df["Monetary"] + 0.1 * campaign_df["LTV"]
        uplift_component = (
            0.2 * campaign_df["Monetary"] + 0.3 * campaign_df["LTV"]
        ) * campaign_df["treatment"]
        noise = rng.normal(0, 0.1 * (base.mean() + 1e-6), size=n)
        campaign_df["outcome"] = (base + uplift_component + noise).clip(lower=0)

        st.dataframe(campaign_df.head())

    if campaign_df is not None:
        # Choose features for uplift model
        possible_feats = [
            c
            for c in campaign_df.columns
            if c
            not in [
                "CustomerID",
                "treatment",
                "outcome",
                "uplift",
            ]
            and np.issubdtype(campaign_df[c].dtype, np.number)
        ]

        default_feats = [
            f
            for f in ["Recency", "Frequency", "Monetary", "LTV", "churn_prob"]
            if f in possible_feats
        ]
        if not default_feats and possible_feats:
            default_feats = possible_feats[:4]

        feature_cols = st.multiselect(
            "Select features for uplift modelling",
            options=possible_feats,
            default=default_feats,
        )

        if len(feature_cols) == 0:
            st.warning("Select at least one numeric feature.")
        elif "treatment" not in campaign_df.columns or "outcome" not in campaign_df.columns:
            st.error(
                "Campaign data must contain 'treatment' and 'outcome' columns."
            )
        else:
            if st.button("Train uplift model and compute uplift scores"):
                with st.spinner("Training DR-Learner uplift model..."):
                    X, T, Y = prepare_uplift_data(
                        campaign_df,
                        treatment_col="treatment",
                        outcome_col="outcome",
                        feature_cols=feature_cols,
                    )
                    uplift_model = train_drl_uplift_model(X, T, Y)
                st.success("Uplift model trained.")

                with st.spinner("Scoring uplift on all customers..."):
                    uplift_scores = predict_uplift(uplift_model, X)
                    uplift_df = uplift_table(campaign_df, uplift_scores)

                st.subheader("Top 20 high-uplift customers")
                st.dataframe(uplift_df.head(20))

                st.subheader("Uplift score distribution")
                fig_u = px.histogram(
                    uplift_scores,
                    nbins=40,
                    title="Distribution of Predicted Uplift",
                )
                st.plotly_chart(fig_u, use_container_width=True)

                st.download_button(
                    "Download full uplift targeting list (CSV)",
                    uplift_df.to_csv(index=False).encode("utf-8"),
                    "uplift_targets.csv",
                    "text/csv",
                )

# ----------------- REALTIME EXAMPLE -----------------
elif page == "Realtime (example)":
    st.markdown(
        "<div class='big-font'>Real-time Inference — Example Scaffolding</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Save these scripts as separate `.py` files and run them outside Streamlit."
    )

    st.subheader("FastAPI scorer example")
    st.code(FASTAPI_SCORER, language="python")

    st.subheader("Producer simulator example")
    st.code(PRODUCER_SIM, language="python")

# ----------------- AI RECOMMENDATIONS -----------------
elif page == "AI Recommendations":

    st.markdown("<div class='big-font'>AI-Driven Campaign Strategy (SHAP-Enhanced)</div>", unsafe_allow_html=True)

    st.write(
        "This upgraded engine uses behavioural SHAP attributions to determine the dominant churn/LTV drivers "
        "for each segment and recommend targeted interventions rooted in model explainability."
    )

    feature_cols = ["Recency", "Frequency", "Monetary"]
    X = rfm[feature_cols]
    X_log = np.log1p(X)
    X_scaled = scaler.transform(X_log)

    # --- Compute SHAP values if not already loaded ---
    @st.cache_resource
    def get_shap_values():
        explainer = shap.Explainer(churn_model, X_scaled)
        shap_vals = explainer(X_scaled)
        return explainer, shap_vals

    try:
        explainer, shap_vals = get_shap_values()
    except Exception as e:
        st.error(f"Cannot compute SHAP values: {e}")
        st.stop()

    # Compute mean absolute SHAP for segments
    shap_df = pd.DataFrame(shap_vals.values, columns=feature_cols)
    shap_df["Segment"] = rfm["Segment"].values

    seg_shap = (
        shap_df.groupby("Segment")[feature_cols]
        .apply(lambda df: df.abs().mean())
        .reset_index()
    )

    st.subheader("Segment-Level SHAP Drivers")
    st.write("Higher |SHAP| means the feature is a stronger behavioural driver for that segment.")

    fig_heat = px.imshow(
        seg_shap.set_index("Segment"),
        text_auto=True,
        color_continuous_scale="Tealrose",
        aspect="auto",
        title="Behavioural Pressure Map (Mean |SHAP| by Segment)"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # -------------------------
    # BUILD SHAP-AWARE RECOMMENDATION ENGINE
    # -------------------------
    def infer_driver_intent(segment_name):
        row = seg_shap[seg_shap["Segment"] == segment_name][feature_cols].iloc[0]
        ranking = row.sort_values(ascending=False)

        primary = ranking.index[0]
        secondary = ranking.index[1]

        return primary, secondary, ranking

    def shap_based_campaign(segment, avg_ltv, avg_churn, ranking):
        primary, secondary = ranking.index[:2]

        rec, freq, mon = ranking.index

        strategies = []

        # 1 — RECENCY is top driver → churn risk is recency-sensitive
        if primary == "Recency":
            strategies.append("Re-engagement flow (recency-driven churn).")
            strategies.append("Trigger push/email/SMS nudges soon after inactivity.")
            strategies.append("Limited-time incentives to shorten the gap between purchases.")

        # 2 — FREQUENCY is top driver → habit strength matters
        if primary == "Frequency":
            strategies.append("Build purchase cadence (increase habit frequency).")
            strategies.append("Introduce subscription, auto-refill or loyalty milestone bonuses.")
            strategies.append("Use personalised replenishment reminders.")

        # 3 — MONETARY is top driver → purchase value sensitivity
        if primary == "Monetary":
            strategies.append("Premiumisation or bundle strategy for high-value cohorts.")
            strategies.append("A/B test average basket-size boosters.")
            strategies.append("Targeted high-value product recommendations.")

        # Add churn level modifiers
        if avg_churn > 0.45:
            strategies.append("Segment is high-risk — activate rescue campaigns immediately.")
        elif avg_churn < 0.15:
            strategies.append("Segment is stable — protect loyalty with exclusives.")

        # Add LTV modifiers
        if avg_ltv > rfm["LTV"].median():
            strategies.append("High-value segment — prioritise personalised experiences.")
        else:
            strategies.append("Mid/low-value segment — allocate budget selectively.")

        return strategies

    # -------------------------
    # BUILD FINAL TABLE
    # -------------------------
    seg_metrics = rfm.groupby("Segment").agg(
        avg_LTV=("LTV", "mean"),
        avg_churn=("churn_prob", "mean"),
        customers=("CustomerID", "count"),
        revenue=("Monetary", "sum"),
    ).reset_index()

    # Merge in SHAP drivers
    seg_metrics = seg_metrics.merge(seg_shap, on="Segment", how="left")

    st.subheader("Segment Intelligence Table (SHAP-Driven)")

    st.dataframe(
        seg_metrics.style.format({
            "avg_LTV": "{:.2f}",
            "avg_churn": "{:.2%}",
            "revenue": "{:.0f}",
            "Recency": "{:.3f}",
            "Frequency": "{:.3f}",
            "Monetary": "{:.3f}",
        })
    )

    st.markdown("## SHAP-Aware Campaign Playbooks")

    for _, row in seg_metrics.iterrows():
        segment = row["Segment"]
        avg_ltv = row["avg_LTV"]
        avg_churn = row["avg_churn"]

        st.markdown(f"### **{segment}**")

        primary, secondary, ranking = infer_driver_intent(segment)

        st.write(f"**Primary behavioural driver:** {primary}")
        st.write(f"**Secondary driver:** {secondary}")

        strategies = shap_based_campaign(segment, avg_ltv, avg_churn, ranking)

        st.markdown("#### Recommended Actions")
        for s in strategies:
            st.write(f"- {s}")

        # ----------------------
        # Dynamic SHAP-based messaging
        # ----------------------
        st.markdown("#### Suggested Messaging (Based on SHAP Root Cause)")

        if primary == "Recency":
            st.code(
                "Subject: We’ve got something new waiting for you\n"
                "Hi {name},\nWe noticed it’s been a while — here’s an early-access preview "
                "and a small thank-you credit if you check in this week."
            )
        elif primary == "Frequency":
            st.code(
                "Subject: Your routine just got easier\n"
                "Hi {name},\nWe thought you'd enjoy simplifying things — here’s a personalised "
                "replenishment reminder and loyalty milestone bonus for staying consistent."
            )
        elif primary == "Monetary":
            st.code(
                "Subject: A premium pick curated just for you\n"
                "Hi {name},\nWe curated a few high-value recommendations that align with what you love — "
                "try them with a limited premium bonus."
            )

        st.markdown("---")

# ----------------- EXPORT -----------------
elif page == "Export":
    st.markdown(
        "<div class='big-font'>Exports & Model Artifacts</div>",
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download Full Segmented Data (CSV)",
        rfm.to_csv(index=False).encode("utf-8"),
        "rfm_segments.csv",
        "text/csv",
    )

    st.download_button(
        "Download Cluster Summary (CSV)",
        cluster_stats.to_csv(index=False).encode("utf-8"),
        "rfm_cluster_summary.csv",
        "text/csv",
    )

    st.markdown("### Save churn model & scaler for real-time scoring")
    if st.button("Export churn model & scaler to ./models"):
        import pickle

        os.makedirs("models", exist_ok=True)
        try:
            pickle.dump(churn_model, open("models/churn_model.pkl", "wb"))
            pickle.dump(scaler, open("models/scaler.pkl", "wb"))
            st.success("Saved churn_model.pkl and scaler.pkl in ./models")
        except Exception as e:
            st.error(f"Failed to save models: {e}")

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown(
    f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Corporate RFM Suite"
)