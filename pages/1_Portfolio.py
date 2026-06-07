import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json

from utils.styles import inject_css, set_page_config
from utils.components import (
    render_health_banner, render_metric_card, render_insight_card,
    render_section_label, render_section_title, risk_badge
)
from utils.charts import risk_donut_chart, recovery_histogram, apply_fintech_theme, plotly_config, model_comparison_chart
from utils.insights import load_metrics, generate_dashboard_insight

set_page_config("Portfolio", icon="")
inject_css()

st.markdown("""
<style>
.kpi-band { display: flex; gap: 10px; margin-bottom: 1.2rem; flex-wrap: wrap; }
.kpi-band > div { flex: 1; min-width: 140px; }
</style>
""", unsafe_allow_html=True)

render_section_label("Overview")
render_section_title("Portfolio Health")

# Load metrics
metrics = load_metrics()
roc_auc = metrics.get('roc_auc', 0.737)

# Portfolio health banner (simulated from current data)
render_health_banner(total_loans=307511, critical_count=87, last_updated="2h ago")

# KPI band
st.markdown('<div class="kpi-band">', unsafe_allow_html=True)
kpi_cols = st.columns(5)
with kpi_cols[0]:
    render_metric_card("Total Borrowers", "307,511", delta="+1.2%", delta_color="normal")
with kpi_cols[1]:
    recovery_rate = (1 - metrics.get('recall', 0.08)) * 100
    render_metric_card("Recovery Rate", f"{recovery_rate:.1f}%", delta="+0.4%", delta_color="normal")
with kpi_cols[2]:
    default_rate = metrics.get('recall', 0.08) * 100
    render_metric_card("Default Rate", f"{default_rate:.1f}%", delta="-0.2%", delta_color="inverse")
with kpi_cols[3]:
    render_metric_card("Portfolio at Risk", "₹4.2Cr", delta="↑ 12L", delta_color="inverse")
with kpi_cols[4]:
    render_metric_card("Model AUC", f"{roc_auc:.3f}", delta="Stable", delta_color="normal")
st.markdown('</div>', unsafe_allow_html=True)

# AI Insight card
render_insight_card(generate_dashboard_insight())

st.markdown("---")

# Charts section
render_section_label("Distribution")
render_section_title("Risk Portfolio Analysis")

chart_left, chart_right = st.columns([2, 1])

with chart_left:
    np.random.seed(42)
    risk_scores = np.concatenate([
        np.random.beta(2, 8, 70000),
        np.random.beta(5, 5, 8000),
        np.random.beta(8, 2, 2000)
    ])
    threshold = metrics.get('threshold', 0.38)
    fig_hist = recovery_histogram(risk_scores, threshold=threshold)
    st.plotly_chart(fig_hist, width='stretch', config=plotly_config)

with chart_right:
    risk_counts = {"Critical": 908, "Low": 2000, "Moderate": 1500, "High": 73592}
    if metrics.get('recall', 0.08) > 0.10:
        risk_counts = {"Critical": 1408, "Low": 3000, "Moderate": 2000, "High": 73592}
    fig_donut = risk_donut_chart(risk_counts)
    st.plotly_chart(fig_donut, width='stretch', config=plotly_config)

st.markdown("---")

# Top critical accounts
render_section_label("Action items")
render_section_title("Top Critical Accounts")

high_risk_data = {
    'Customer_ID': ['C10001', 'C10045', 'C10234', 'C10567', 'C10890', 'C11023', 'C11245', 'C11567', 'C11890', 'C12001'],
    'Risk_Probability': [0.85, 0.78, 0.72, 0.68, 0.65, 0.62, 0.59, 0.57, 0.55, 0.52],
    'Loan_Amount': [450000, 320000, 580000, 210000, 670000, 390000, 520000, 280000, 610000, 340000],
    'Debt_Ratio': [0.82, 0.75, 0.71, 0.68, 0.64, 0.61, 0.58, 0.55, 0.53, 0.50],
    'Top_Risk_Factor': ['High Debt', 'Low Income', 'Unstable Job', 'High Inquiries', 'No Collateral',
                        'High Debt', 'Unstable Job', 'Low Income', 'High Inquiries', 'No Collateral'],
    'Action': ['Escalate', 'Call', 'Escalate', 'Call', 'Review',
               'Call', 'Review', 'Call', 'Review', 'Call']
}
high_risk_df = pd.DataFrame(high_risk_data)

# Color-code risk probability column
styled = high_risk_df.style.map(
    lambda v: "color: #E24B4A; font-weight: 500;" if v >= 0.70 else "color: #D4A24C;" if v >= 0.50 else "color: #4A9FE6;",
    subset=["Risk_Probability"]
).format({"Risk_Probability": "{:.0%}", "Debt_Ratio": "{:.2f}", "Loan_Amount": "{:,.0f}"})
st.dataframe(styled, width='stretch')

st.markdown("---")

# Alerts section
render_section_label("Alerts")
render_section_title("Risk Alerts")

alert_cols = st.columns(3)
with alert_cols[0]:
    st.error("High-Risk Loan Exposure Alert — 3 new critical accounts since last login")
with alert_cols[1]:
    st.warning("Increasing Default Trend Alert — default rate up 0.3% this week")
with alert_cols[2]:
    st.success("Model Performance Stable — AUC holding at 0.737")

st.markdown("---")

# Model comparison
model_comparison_path = Path("artifacts/model_trainer/model_comparison.json")
if model_comparison_path.exists():
    with open(model_comparison_path) as f:
        model_comparison = json.load(f)
    if model_comparison:
        render_section_label("Models")
        render_section_title("Model Comparison")
        comparison_df = pd.DataFrame(model_comparison).T.reset_index().rename(columns={'index': 'Model'})
        fig_comp = model_comparison_chart(comparison_df)
        st.plotly_chart(fig_comp, width='stretch', config=plotly_config)

st.markdown("---")
st.caption("This dashboard provides AI-driven insights for decision support. Final loan decisions should include business review and human verification.")
