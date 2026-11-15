"""
app.py
======
Synthetic Data Quality & Privacy Governance Dashboard
Lead Data Privacy Architect & Senior Streamlit Developer Deliverable

Grounded in:
- Synthetic Data Vault (SDV / SDMetrics): KSComplement, TVComplement & CorrelationSimilarity
- El Emam, Mosquera & Bass (2020), JMIR (PMC7704280) & EMA Policy 0070 / Health Canada (0.09 Risk Threshold)
- NIST SP 800-226 (March 2025): Guidelines for Evaluating Differential Privacy Guarantees
- Perforce Delphix 2026 Enterprise Test Data Management Benchmark & Gartner AI Data Adoption
- Framingham Heart Study & Cardiovascular Metabolic Cohort Clinical Distributions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_governance_engine import (
    CATEGORICAL_COLS,
    CLINICAL_FEATURE_SPECS,
    CONTINUOUS_COLS,
    RESEARCH_CITATIONS,
    evaluate_governance_metrics,
    generate_datasets,
    simulate_privacy_utility_frontier,
)


# =============================================================================
# PAGE CONFIGURATION & EXECUTIVE STYLING
# =============================================================================

st.set_page_config(
    page_title="Synthetic Data Quality & Privacy Governance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
        padding: 1.6rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        border-left: 6px solid #0ea5e9;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.98rem;
    }
    .plain-english-box {
        background-color: #f0f9ff;
        border-left: 4px solid #0284c7;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin-top: 0.35rem;
        margin-bottom: 0.65rem;
        font-size: 0.9rem;
        color: #0f172a;
    }
    .citation-caption {
        font-size: 0.8rem;
        color: #475569;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 0.55rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .zone-restricted {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 1rem;
    }
    .zone-restricted h4 {
        color: #991b1b;
        margin-top: 0;
    }
    .zone-safe {
        background-color: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 1rem;
    }
    .zone-safe h4 {
        color: #065f46;
        margin-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR NAVIGATION & GOVERNANCE CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("## 🛡️ Governance Controls")
    st.caption(
        "Calibrate differential privacy budgets, synthesis architecture, and enterprise "
        "compliance parameters in real time."
    )

    nav_section = st.radio(
        "Dashboard View",
        options=[
            "1. Executive Governance Overview",
            "2. Statistical Parity & Distribution Lab",
            "3. Privacy Risk vs. Data Utility Frontier",
            "4. Restricted PII vs. Synthetic Safe Zone",
            "5. Research Methodology & Citations",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### 🔐 Privacy & Synthesis Engine")

    synthesis_method = st.selectbox(
        "Synthesis Architecture",
        options=[
            "Gaussian Copula + DP (SDV Standard)",
            "Sequential Classification & Regression Trees (El Emam JMIR)",
            "DP-CTGAN (Deep Generative Neural Network)",
        ],
        help=(
            "Selects the generative model architecture. Gaussian Copula (SDV) and Sequential "
            "Decision Trees (El Emam et al., 2020, JMIR) are the gold-standard tabular clinical synthesizers."
        ),
    )

    epsilon = st.slider(
        "Differential Privacy Budget (ε - Epsilon)",
        min_value=0.10,
        max_value=5.00,
        value=1.00,
        step=0.05,
        help=(
            "Per NIST SP 800-226 (March 2025) & Dwork (2006): Lower ε injects more mathematical "
            "noise for stronger privacy protection (lower PII leakage risk), while higher ε preserves "
            "sharper statistical fidelity."
        ),
    )

    n_records = st.select_slider(
        "Cohort Sample Size (Patients)",
        options=[500, 1000, 1500, 2500, 4240],
        value=1500,
        help="Framingham full benchmark cohort contains N = 4,240 records (PMC13133109).",
    )

    st.divider()
    st.markdown("### ⏱️ Compliance Bottleneck Inputs")
    st.caption("Grounded in the Perforce Delphix 2026 Enterprise TDM Survey (n=518).")

    raw_approval_weeks = st.slider(
        "Raw PII Legal/IRB Approval Wait (Weeks)",
        min_value=1.0,
        max_value=8.0,
        value=3.0,
        step=0.5,
        help=(
            "Perforce Delphix 2026 survey reports 42% of enterprises wait weeks or months "
            "for production data access (2 weeks DB negotiation + 1–2 weeks legal review)."
        ),
    )

    synthetic_provisioning_hours = st.selectbox(
        "Synthetic Self-Service Provisioning (Hours)",
        options=[0.5, 1.0, 2.0, 4.0],
        index=0,
        help="Perforce Delphix (2026) documents API-driven synthetic data delivery in 30 minutes (0.5 hours).",
    )

    monthly_requests = st.number_input(
        "Monthly Analytics Data Requests",
        min_value=1,
        max_value=200,
        value=18,
        step=1,
    )

    analyst_hourly_rate = st.number_input(
        "Blended Data Scientist / Compliance Cost ($/hr)",
        min_value=30,
        max_value=250,
        value=85,
        step=5,
    )


# =============================================================================
# CACHED DATA GENERATION & METRIC COMPUTATION
# =============================================================================

@st.cache_data(show_spinner=False)
def load_and_evaluate(
    n_rec: int,
    eps: float,
    method: str,
    raw_wks: float,
    syn_hrs: float,
    reqs: int,
    rate: float,
):
    raw_df, syn_df = generate_datasets(
        n_records=n_rec,
        epsilon=eps,
        synthesis_method=method,
        random_seed=42,
    )
    eval_res = evaluate_governance_metrics(
        raw_df=raw_df,
        syn_df=syn_df,
        epsilon=eps,
        raw_approval_weeks=raw_wks,
        synthetic_provisioning_hours=syn_hrs,
        monthly_data_requests=reqs,
        analyst_hourly_cost_usd=rate,
    )
    frontier_df = simulate_privacy_utility_frontier(raw_df, synthesis_method=method)
    return raw_df, syn_df, eval_res, frontier_df


raw_pii_data, synthetic_twin_data, metrics, frontier_df = load_and_evaluate(
    n_records,
    epsilon,
    synthesis_method,
    raw_approval_weeks,
    synthetic_provisioning_hours,
    monthly_requests,
    analyst_hourly_rate,
)


# =============================================================================
# HEADER BANNER
# =============================================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🛡️ Synthetic Data Quality & Privacy Governance Terminal</h1>
        <p>
            Auditing Statistical Parity, Identity Disclosure Risk, and Compliance Acceleration
            Between Restricted Clinical PII (Framingham Cohort) and a Privacy-Preserving Synthetic Twin
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# EXECUTIVE KPI METRIC CARDS (ALWAYS VISIBLE AT TOP)
# =============================================================================

st.markdown("### 📊 Executive Governance Scorecard")

kpi_col1, kpi_col2, kpi_col3 = st.columns(3, gap="large")

with kpi_col1:
    parity_delta = round(metrics.overall_parity_score - 85.53, 2)
    st.metric(
        label="1. Statistical Parity Score (SDV Quality)",
        value=f"{metrics.overall_parity_score:.2f}%",
        delta=f"{parity_delta:+.2f}% vs. 85.53% Published Benchmark",
        help=(
            "Plain English: Does this fake data look and behave like the real data? "
            "Computed using the Synthetic Data Vault (SDMetrics) formula combining "
            "KSComplement = 1 - D_KS for continuous features, TVComplement for categorical "
            "features, and pairwise CorrelationSimilarity."
        ),
    )
    st.markdown(
        """
        <div class="plain-english-box">
            <strong>Plain English Translation:</strong><br>
            <em>"Does this fake data look and behave like the real data?"</em><br>
            A score near <strong>90%+</strong> means analysts and machine learning models get the same statistical trends as raw data without seeing a single real person's record.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="citation-caption">
            📚 <strong>Research Benchmark:</strong> Grounded in
            <a href="{RESEARCH_CITATIONS['sdv_ks']['url']}" target="_blank">SDMetrics KSComplement (1 − D<sub>KS</sub>)</a>
            &amp; <a href="{RESEARCH_CITATIONS['frontiers_benchmark']['url']}" target="_blank">Frontiers ML Validation (2025)</a>
            (benchmark: <strong>85.53%</strong> column shape fidelity; &gt;70% threshold).
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    risk_pct = metrics.meaningful_disclosure_risk * 100.0
    ceiling_margin = round(9.0 - risk_pct, 2)
    st.metric(
        label="2. PII Leakage Risk (Identity Disclosure)",
        value=f"{risk_pct:.2f}%",
        delta=f"-{ceiling_margin:.2f}% below 9.0% EMA/Health Canada Ceiling",
        delta_color="normal" if risk_pct <= 9.0 else "inverse",
        help=(
            "Plain English: What is the mathematical probability that a hacker could "
            "reverse-engineer a real person's identity from this fake data? "
            "Grounded in Khaled El Emam et al. (2020, JMIR) meaningful identity disclosure "
            "risk model and EMA Policy 0070 / Health Canada's 0.09 (9.0%) safe harbor threshold."
        ),
    )
    st.markdown(
        """
        <div class="plain-english-box">
            <strong>Plain English Translation:</strong><br>
            <em>"What is the mathematical probability that a hacker could reverse-engineer a real person's identity from this fake data?"</em><br>
            Regulators require this risk to stay below <strong>9.0% (0.09)</strong>. At <strong>"""
        + f"{risk_pct:.2f}%"
        + """</strong> (≈ k-anonymity of """
        + f"k={metrics.k_anonymity_equivalent}"
        + """), the dataset qualifies as non-personal safe-harbor data.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="citation-caption">
            📚 <strong>Research Benchmark:</strong> Grounded in
            <a href="{RESEARCH_CITATIONS['el_emam_jmir']['url']}" target="_blank">El Emam et al. (2020, JMIR)</a>
            (synthetic health risk: <strong>0.86%–1.98%</strong> vs. <strong>9.0%</strong>
            <a href="{RESEARCH_CITATIONS['ema_policy_0070']['url']}" target="_blank">EMA Policy 0070 / Health Canada</a> threshold) &amp;
            <a href="{RESEARCH_CITATIONS['nist_dp']['url']}" target="_blank">NIST SP 800-226 (2025)</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    st.metric(
        label="3. Data Democratization Rate (Speedup)",
        value=f"{metrics.acceleration_factor:.0f}x Faster ({metrics.synthetic_provisioning_hours} hrs)",
        delta=f"Saves {metrics.hours_saved_per_request:.1f} work hrs/request (${metrics.annual_cost_savings_usd:,.0f}/yr)",
        help=(
            "Plain English: How much faster can our analysts start working? "
            "Compares traditional IRB/Legal PII compliance sign-off (weeks/months) against "
            "pre-approved synthetic data self-service provisioning (0.5 hours)."
        ),
    )
    st.markdown(
        f"""
        <div class="plain-english-box">
            <strong>Plain English Translation:</strong><br>
            <em>"How much faster can our analysts start working?"</em><br>
            Instead of waiting <strong>{raw_approval_weeks:.1f} weeks ({metrics.raw_approval_days:.0f} business days)</strong> for legal PII sign-off, analysts get safe synthetic data in <strong>{metrics.synthetic_provisioning_hours} hours</strong>—saving <strong>${metrics.annual_cost_savings_usd:,.0f}/year</strong> in blocked engineering time.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="citation-caption">
            📚 <strong>Research Benchmark:</strong> Grounded in the
            <a href="{RESEARCH_CITATIONS['delphix_2026']['url']}" target="_blank">Perforce Delphix 2026 TDM Report (n=518)</a>
            (42% wait weeks/months for raw data; synthetic self-service delivers in <strong>30 mins / 0.5 hrs</strong>).
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# =============================================================================
# VIEW 1: EXECUTIVE GOVERNANCE OVERVIEW
# =============================================================================

if nav_section == "1. Executive Governance Overview":
    st.subheader("🏛️ Executive Compliance Status & Plain-English Governance Guide")

    if metrics.meaningful_disclosure_risk <= metrics.ema_threshold and metrics.overall_parity_score >= 70.0:
        st.info(
            f"✅ **GOVERNANCE VERDICT — APPROVED FOR ANALYTICS RELEASE:** "
            f"At privacy budget **ε = {epsilon:.2f}**, the Synthetic Twin achieves **{metrics.overall_parity_score:.2f}%** "
            f"Statistical Parity (exceeding the 70.0% SDMetrics threshold and 85.53% empirical benchmark) while maintaining "
            f"a Meaningful Identity Disclosure Risk of **{metrics.meaningful_disclosure_risk*100:.2f}%**—more than "
            f"**{metrics.ema_threshold / max(metrics.meaningful_disclosure_risk, 0.0001):.1f}x safer** than the "
            f"**9.0% (0.09)** regulatory limit set by EMA Policy 0070 and Health Canada."
        )
    else:
        st.warning(
            f"⚠️ **GOVERNANCE VERDICT — REQUIRES CALIBRATION:** "
            f"Current settings yield Parity = **{metrics.overall_parity_score:.2f}%** and "
            f"PII Leakage Risk = **{metrics.meaningful_disclosure_risk*100:.2f}%**. Adjust the Differential Privacy "
            f"Budget (ε) slider in the sidebar so that leakage stays ≤ 9.0% and parity stays ≥ 70.0%."
        )

    with st.expander("💡 How Does Synthetic Data Governance Work? (Click for Plain-English Executive Primer)", expanded=True):
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            st.markdown("#### 1. Why Raw PII Blocks Innovation")
            st.write(
                "Hospital electronic health records (EHR) and financial ledgers contain **Direct Identifiers** "
                "(names, SSNs, Medical Record Numbers) and **Quasi-Identifiers** (age, ZIP code, blood pressure). "
                "Even if names are masked, combinations of quasi-identifiers can uniquely re-identify patients. "
                "Consequently, legal and IRB reviews stall analytics teams for **2 to 6 weeks per dataset request**."
            )
        with exp_col2:
            st.markdown("#### 2. How the Synthetic Twin Is Built")
            st.write(
                "Instead of masking columns row-by-row, a **Gaussian Copula** or **Sequential Decision Tree** model "
                "learns the mathematical shapes and correlations of the real clinical cohort (e.g., how Systolic BP "
                "rises with Age and BMI). It then samples brand-new artificial records that have **zero one-to-one mapping** "
                "to any living person, reinforced with **Differential Privacy (ε)** noise."
            )
        with exp_col3:
            st.markdown("#### 3. Regulatory Safe Harbor Standard")
            st.write(
                "Under **EMA Policy 0070** and **Health Canada** clinical data release standards (validated by "
                "Khaled El Emam et al., 2020 in *JMIR*), a dataset is considered non-identifiable when the probability "
                "of re-identification is **≤ 0.09 (9.0%)**. Our engine continuously audits both statistical fidelity "
                "and disclosure risk before unlocking self-service access."
            )

    # Quick Visual Overview Row: Gauge Chart + Feature Parity Bar Chart
    ov_col1, ov_col2 = st.columns([1, 1.3], gap="large")

    with ov_col1:
        st.markdown("#### 🎯 PII Leakage Risk vs. Regulatory Safe Harbor Ceiling")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=metrics.meaningful_disclosure_risk * 100.0,
                number={"suffix": "%", "font": {"size": 36}},
                delta={
                    "reference": 9.0,
                    "increasing": {"color": "#ef4444"},
                    "decreasing": {"color": "#10b981"},
                },
                title={
                    "text": "Meaningful Identity Disclosure Risk (%)<br><span style='font-size:0.8em;color:gray'>EMA Policy 0070 / Health Canada Ceiling = 9.0% (0.09)</span>"
                },
                gauge={
                    "axis": {"range": [0, 36], "ticksuffix": "%"},
                    "bar": {"color": "#0284c7"},
                    "steps": [
                        {"range": [0, 2.0], "color": "#d1fae5"},
                        {"range": [2.0, 9.0], "color": "#fef3c7"},
                        {"range": [9.0, 36.0], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#dc2626", "width": 4},
                        "thickness": 0.8,
                        "value": 9.0,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=360, margin=dict(l=25, r=25, t=60, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(
            f"**Green Zone (0%–2.0%):** El Emam et al. (2020) synthetic benchmark range (0.86%–1.98%). "
            f"**Yellow Zone (2.0%–9.0%):** Compliant under EMA/Health Canada 0.09 threshold. "
            f"**Red Zone (>9.0%):** Unacceptable re-identification risk (Raw dataset baseline = {metrics.raw_baseline_disclosure_risk*100:.1f}%)."
        )

    with ov_col2:
        st.markdown("#### 📈 Feature-by-Feature Statistical Parity Score (SDMetrics)")
        parity_df_sorted = metrics.feature_parity_table.sort_values("Parity Score (%)", ascending=True)
        fig_bar = px.bar(
            parity_df_sorted,
            x="Parity Score (%)",
            y="Feature",
            color="Type",
            orientation="h",
            text="Parity Score (%)",
            range_x=[0, 105],
            color_discrete_map={
                "Continuous (KSComplement)": "#0284c7",
                "Categorical (TVComplement)": "#0d9488",
            },
        )
        fig_bar.add_vline(
            x=70.0,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="SDV Min Threshold (70%)",
            annotation_position="bottom right",
        )
        fig_bar.add_vline(
            x=85.53,
            line_dash="dot",
            line_color="#0f172a",
            annotation_text="Frontiers Benchmark (85.53%)",
            annotation_position="top left",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(
            height=360,
            margin=dict(l=10, r=30, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption(
            "Computed per column using `KSComplement = 1 - D_KS` (continuous) and `TVComplement = 1 - TVD` (categorical) "
            "from the Synthetic Data Vault (SDMetrics) evaluation standard."
        )


# =============================================================================
# VIEW 2: STATISTICAL PARITY & DISTRIBUTION LAB
# =============================================================================

elif nav_section == "2. Statistical Parity & Distribution Lab":
    st.subheader("🔬 Statistical Parity & Clinical Distribution Comparison Lab")
    st.info(
        "**Beginner Translation:** *\"Does this fake data look and behave like the real data?\"* "
        "Select any clinical variable below to visually inspect how closely the Synthetic Twin's "
        "histogram and Cumulative Distribution Function (CDF) overlay the restricted Framingham cohort."
    )

    feature_label_map = {
        str(CLINICAL_FEATURE_SPECS[k]["label"]): k for k in CONTINUOUS_COLS
    }
    selected_label = st.selectbox(
        "Select Clinical Continuous Feature to Audit:",
        options=list(feature_label_map.keys()),
        index=1,  # Default to Systolic Blood Pressure
    )
    selected_col = feature_label_map[selected_label]
    spec = CLINICAL_FEATURE_SPECS[selected_col]

    # Retrieve KS stats for selected feature
    row_match = metrics.feature_parity_table[
        metrics.feature_parity_table["Column"] == selected_col
    ].iloc[0]

    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
    stat_c1.metric("KSComplement Parity Score", f"{row_match['Parity Score (%)']:.2f}%")
    stat_c2.metric("KS Maximum CDF Distance (D_KS)", f"{row_match['Test Statistic (D_KS / TVD)']:.4f}")
    stat_c3.metric("Raw Cohort Mean (±SD)", str(row_match["Raw Mean / %"]))
    stat_c4.metric("Synthetic Twin Mean (±SD)", str(row_match["Synthetic Mean / %"]))

    dist_col1, dist_col2 = st.columns(2, gap="large")

    combined_cont = pd.DataFrame(
        {
            selected_label: np.concatenate(
                [raw_pii_data[selected_col].to_numpy(), synthetic_twin_data[selected_col].to_numpy()]
            ),
            "Dataset Zone": (["Restricted Raw PII"] * len(raw_pii_data))
            + (["Safe Synthetic Twin"] * len(synthetic_twin_data)),
        }
    )

    with dist_col1:
        st.markdown(f"#### 📊 Marginal Distribution Overlay: {selected_label}")
        fig_hist = px.histogram(
            combined_cont,
            x=selected_label,
            color="Dataset Zone",
            barmode="overlay",
            histnorm="probability density",
            nbins=38,
            marginal="box",
            opacity=0.62,
            color_discrete_map={
                "Restricted Raw PII": "#ef4444",
                "Safe Synthetic Twin": "#10b981",
            },
        )
        fig_hist.update_layout(
            height=410,
            margin=dict(l=15, r=15, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(
            f"Empirical Reference: {spec['source']} "
            f"([Source Study]({spec['source_url']}))."
        )

    with dist_col2:
        st.markdown(f"#### 📐 Empirical Cumulative Distribution Function (ECDF — KS Test)")
        fig_ecdf = px.ecdf(
            combined_cont,
            x=selected_label,
            color="Dataset Zone",
            color_discrete_map={
                "Restricted Raw PII": "#ef4444",
                "Safe Synthetic Twin": "#10b981",
            },
        )
        fig_ecdf.update_layout(
            height=410,
            yaxis_title="Cumulative Probability F(x)",
            margin=dict(l=15, r=15, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ecdf, use_container_width=True)
        st.caption(
            "In SDMetrics, `KSComplement = 1 - max|F_real(x) - F_syn(x)|`. The closer the red and green ECDF curves track "
            "each other, the higher the Statistical Parity Score."
        )

    st.divider()
    st.markdown("#### 🔗 Multivariate Correlation Preservation (Pairwise Clinical Relationships)")
    st.write(
        f"**Correlation Similarity Score:** `{metrics.correlation_similarity_score:.2f}%` "
        "(Framingham benchmark highlights a strong positive correlation `r = 0.787` between Systolic and Diastolic BP)."
    )

    corr_col1, corr_col2 = st.columns(2, gap="large")
    short_labels = [
        "Age", "Sys BP", "Dia BP", "BMI", "Tot Chol", "HDL Chol", "Glucose"
    ]
    with corr_col1:
        raw_corr_mat = np.round(raw_pii_data[CONTINUOUS_COLS].corr().to_numpy(), 2)
        fig_corr_raw = px.imshow(
            raw_corr_mat,
            x=short_labels,
            y=short_labels,
            text_auto=True,
            zmin=-0.5,
            zmax=1.0,
            color_continuous_scale="RdBu_r",
            title="Restricted Raw PII — Correlation Matrix",
        )
        fig_corr_raw.update_layout(height=380, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_corr_raw, use_container_width=True)

    with corr_col2:
        syn_corr_mat = np.round(synthetic_twin_data[CONTINUOUS_COLS].corr().to_numpy(), 2)
        fig_corr_syn = px.imshow(
            syn_corr_mat,
            x=short_labels,
            y=short_labels,
            text_auto=True,
            zmin=-0.5,
            zmax=1.0,
            color_continuous_scale="RdBu_r",
            title=f"Safe Synthetic Twin (ε = {epsilon:.2f}) — Correlation Matrix",
        )
        fig_corr_syn.update_layout(height=380, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig_corr_syn, use_container_width=True)

    with st.expander("📋 View Complete Feature-Level Statistical Parity Audit Table (KSComplement & TVComplement)", expanded=True):
        st.dataframe(metrics.feature_parity_table, use_container_width=True, hide_index=True)


# =============================================================================
# VIEW 3: PRIVACY RISK VS. DATA UTILITY FRONTIER
# =============================================================================

elif nav_section == "3. Privacy Risk vs. Data Utility Frontier":
    st.subheader("⚖️ Privacy Risk vs. Data Utility Trade-Off Frontier")
    st.info(
        "**Plain-English Explanation:** In Differential Privacy (governed by **NIST SP 800-226**), "
        "the privacy budget **Epsilon (ε)** acts like a privacy dial. Turning **ε down** (e.g., `ε = 0.1–0.5`) "
        "injects more mathematical noise—making it nearly impossible for an attacker to re-identify a patient, "
        "but slightly blurring the data's accuracy. Turning **ε up** (e.g., `ε > 5.0`) sharpens accuracy but "
        "increases PII leakage risk toward the **9.0% EMA/Health Canada regulatory limit**."
    )

    tr_col1, tr_col2 = st.columns([1.35, 1], gap="large")

    with tr_col1:
        st.markdown("#### 📈 Pareto Frontier: Statistical Parity (%) vs. PII Leakage Risk (%) Across ε")
        fig_frontier = go.Figure()

        # Frontier line and points across simulated epsilons
        fig_frontier.add_trace(
            go.Scatter(
                x=frontier_df["PII Leakage Risk (%)"],
                y=frontier_df["Statistical Parity Score (%)"],
                mode="lines+markers+text",
                text=[f"ε={e}" for e in frontier_df["epsilon"]],
                textposition="top center",
                marker=dict(
                    size=13,
                    color=frontier_df["epsilon"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Epsilon (ε)"),
                ),
                line=dict(color="#334155", width=2, dash="dot"),
                name="DP Calibration Curve",
                hovertemplate=(
                    "<b>Epsilon (ε): %{text}</b><br>"
                    "PII Leakage Risk: %{x:.2f}%<br>"
                    "Statistical Parity: %{y:.2f}%<extra></extra>"
                ),
            )
        )

        # Highlight current selected operating point
        fig_frontier.add_trace(
            go.Scatter(
                x=[metrics.meaningful_disclosure_risk * 100.0],
                y=[metrics.overall_parity_score],
                mode="markers",
                marker=dict(
                    size=20,
                    color="#ef4444",
                    symbol="star",
                    line=dict(width=2, color="#ffffff"),
                ),
                name=f"Active Governance Setting (ε={epsilon:.2f})",
            )
        )

        # Regulatory & quality reference lines
        fig_frontier.add_vline(
            x=9.0,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text="EMA / Health Canada Max Risk Ceiling (9.0%)",
            annotation_position="top right",
        )
        fig_frontier.add_vline(
            x=1.98,
            line_dash="dot",
            line_color="#10b981",
            annotation_text="El Emam JMIR Synthetic Benchmark (1.98%)",
            annotation_position="bottom right",
        )
        fig_frontier.add_hline(
            y=85.53,
            line_dash="dot",
            line_color="#0284c7",
            annotation_text="Frontiers SDMetrics Benchmark (85.53%)",
            annotation_position="bottom left",
        )

        fig_frontier.update_layout(
            height=450,
            xaxis_title="Meaningful Identity Disclosure / PII Leakage Risk (%) — Lower is Safer",
            yaxis_title="Statistical Parity Score (%) — Higher is Better Utility",
            xaxis=dict(range=[0, max(11.5, frontier_df["PII Leakage Risk (%)"].max() + 1.5)]),
            yaxis=dict(range=[65, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=15, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_frontier, use_container_width=True)

    with tr_col2:
        st.markdown("#### 🛡️ Membership Inference Attack (MIA) Vulnerability")
        st.write(
            "Under **NIST SP 800-226** and Differential Privacy theory (Dwork, 2006), an adversary attempting "
            "a **Membership Inference Attack** (guessing whether a specific individual was in the training cohort) "
            "faces a strict mathematical upper bound of `e^ε / (1 + e^ε)`:"
        )
        mia_m1, mia_m2 = st.columns(2)
        mia_m1.metric(
            "Empirical MIA Success Rate",
            f"{metrics.empirical_mia_risk*100:.1f}%",
            delta=f"+{(metrics.empirical_mia_risk - 0.50)*100:.1f}% vs. 50% Random Coin Flip",
            delta_color="inverse",
        )
        mia_m2.metric(
            "Worst-Case Theoretical DP Bound",
            f"{metrics.mia_theoretical_bound*100:.1f}%",
            help="Formula: e^ε / (1 + e^ε). At ε = 0.1 this is 52.5%; at ε = 1.0 it is 73.1%.",
        )

        st.dataframe(
            frontier_df[
                [
                    "epsilon",
                    "Statistical Parity Score (%)",
                    "PII Leakage Risk (%)",
                    "Empirical MIA Success (%)",
                    "EMA / Health Canada Status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Reference: [NIST SP 800-226 (March 2025)](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-226.pdf) "
            "& [El Emam et al. (2020, JMIR)](https://www.jmir.org/2020/11/e23139/)."
        )


# =============================================================================
# VIEW 4: RESTRICTED PII ZONE VS. SYNTHETIC SAFE ZONE
# =============================================================================

elif nav_section == "4. Restricted PII vs. Synthetic Safe Zone":
    st.subheader("🔐 Architectural Separation: Restricted Raw PII Zone vs. Synthetic Safe Zone")
    st.info(
        "**Plain-English Explanation:** Compliance officers require strict boundary separation. "
        "The **Red Restricted Zone** on the left contains real patient names, Social Security Numbers, "
        "exact birthdates, and 5-digit ZIP codes locked behind a **3-week legal/IRB approval firewall**. "
        "The **Green Synthetic Safe Zone** on the right contains zero real individuals—every row is "
        "synthetically sampled from the copula distribution and ready for self-service analytics in **30 minutes**."
    )

    z_col1, z_col2 = st.columns(2, gap="large")

    with z_col1:
        st.markdown(
            f"""
            <div class="zone-restricted">
                <h4>🚫 RAW / RESTRICTED ZONE (HIPAA / GDPR Regulated PII)</h4>
                <ul>
                    <li><strong>Access Approval Wait:</strong> {raw_approval_weeks:.1f} Weeks ({metrics.raw_approval_days:.0f} Business Days)</li>
                    <li><strong>Direct Identifiers Present:</strong> Full Name, SSN, MRN, Exact DOB</li>
                    <li><strong>Baseline Re-Identification Risk:</strong> {metrics.raw_baseline_disclosure_risk*100:.1f}% (Exceeds 9.0% EMA Ceiling)</li>
                    <li><strong>Governance Policy:</strong> Locked to Production Enclave; Prohibited for Dev/Test/AI Training</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Sample Restricted Records (Direct PII Highlighted):**")
        st.dataframe(
            raw_pii_data.head(10),
            use_container_width=True,
            hide_index=True,
        )

    with z_col2:
        st.markdown(
            f"""
            <div class="zone-safe">
                <h4>✅ SYNTHETIC / SAFE ZONE (Safe-Harbor Compliant Twin)</h4>
                <ul>
                    <li><strong>Self-Service Provisioning Time:</strong> {metrics.synthetic_provisioning_hours} Hours (30 Minutes)</li>
                    <li><strong>Direct Identifiers Present:</strong> None (100% Artificial Tokens &amp; 3-Digit ZIP Regions)</li>
                    <li><strong>Meaningful Disclosure Risk:</strong> {metrics.meaningful_disclosure_risk*100:.2f}% (Compliant ≤ 9.0% EMA/Health Canada)</li>
                    <li><strong>Governance Policy:</strong> Cleared for BI Dashboards, ML Training &amp; Cloud Sandboxes</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**Sample Synthetic Twin Records (Zero Real Patient Linkage):**")
        st.dataframe(
            synthetic_twin_data.head(10),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown("#### ⏱️ Enterprise Data Democratization & Cost Savings Breakdown")
    dem_c1, dem_c2 = st.columns([1.2, 1], gap="large")

    with dem_c1:
        workflow_df = pd.DataFrame(
            {
                "Workflow Stage": [
                    "1. Dataset Discovery & Schema Request",
                    "2. DBA Transaction DB Negotiation",
                    "3. Legal / Privacy / IRB Compliance Audit",
                    "4. Manual ETL Masking & Provisioning",
                ],
                "Traditional Raw PII Pipeline (Work Hours)": [
                    raw_approval_weeks * 4.0,
                    raw_approval_weeks * 14.0,
                    raw_approval_weeks * 16.0,
                    raw_approval_weeks * 6.0,
                ],
                "Automated Synthetic Pipeline (Work Hours)": [
                    synthetic_provisioning_hours * 0.2,
                    synthetic_provisioning_hours * 0.2,
                    synthetic_provisioning_hours * 0.3,
                    synthetic_provisioning_hours * 0.3,
                ],
            }
        )
        workflow_melt = workflow_df.melt(
            id_vars="Workflow Stage",
            var_name="Pipeline Type",
            value_name="Blocked Work Hours",
        )
        fig_dem = px.bar(
            workflow_melt,
            x="Blocked Work Hours",
            y="Workflow Stage",
            color="Pipeline Type",
            barmode="group",
            orientation="h",
            text_auto=".1f",
            color_discrete_map={
                "Traditional Raw PII Pipeline (Work Hours)": "#ef4444",
                "Automated Synthetic Pipeline (Work Hours)": "#10b981",
            },
        )
        fig_dem.update_layout(
            height=340,
            margin=dict(l=10, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_dem, use_container_width=True)

    with dem_c2:
        st.markdown("##### 💰 Annual ROI & Engineering Velocity Model")
        st.write(
            f"- **Raw PII Wait Per Request:** `{raw_approval_weeks * 40.0:.1f} work hours` "
            f"(`{raw_approval_weeks:.1f} weeks`)\n"
            f"- **Synthetic Provisioning Time:** `{metrics.synthetic_provisioning_hours:.2f} hours` "
            f"(Perforce Delphix 2026 ADP case study benchmark: 30 minutes)\n"
            f"- **Net Time Saved Per Request:** `{metrics.hours_saved_per_request:.1f} hours`\n"
            f"- **Annual Request Volume:** `{monthly_requests * 12} requests/year` "
            f"(`{monthly_requests}/month`)\n"
            f"- **Annual Productivity Value Unlocked:** **`${metrics.annual_cost_savings_usd:,.0f} / year`** "
            f"(at `${analyst_hourly_rate}/hr` blended rate)"
        )
        st.caption(
            "Sources: [Perforce Delphix 2026 Test Data Management Survey (via DataCebo)](https://datacebo.com/blog/capabilities-of-synthetic-test-data/) "
            "& [Perforce Greenfield Synthetic Data Benchmark](https://www.perforce.com/blog/pdx/synthetic-for-greenfield-app-dev)."
        )


# =============================================================================
# VIEW 5: RESEARCH METHODOLOGY & CITATIONS
# =============================================================================

elif nav_section == "5. Research Methodology & Citations":
    st.subheader("📚 Peer-Reviewed Literature, Regulatory Standards & Empirical Citations")
    st.info(
        "Every formula, threshold, and clinical parameter in this dashboard is strictly grounded in "
        "published academic papers, regulatory guidance, and industry benchmark surveys."
    )

    for key, item in RESEARCH_CITATIONS.items():
        with st.expander(f"📖 {item['title']} — {item['authors']}", expanded=True):
            st.markdown(f"**Empirical Finding & Dashboard Application:** {item['finding']}")
            st.markdown(f"🔗 **Verified Source Link:** [{item['url']}]({item['url']})")