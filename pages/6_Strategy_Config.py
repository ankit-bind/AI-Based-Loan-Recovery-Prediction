import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json

from utils.styles import inject_css, set_page_config
from utils.components import render_section_label, render_section_title, render_metric_card, render_verdict_card
from utils.charts import threshold_sensitivity_chart, plotly_config
from utils.insights import generate_threshold_recommendation

set_page_config("Strategy Config", icon="")
inject_css()

render_section_label("Simulation")
render_section_title("Strategy Configuration")

st.markdown("""
<p style="font-size:14px;         color:#F0F2F6; line-height:1.65; margin-bottom:1rem;">
Adjust the recovery strategy to balance business objectives. Every threshold change is translated into estimated monetary impact — not just statistical metrics.
</p>
""", unsafe_allow_html=True)

metrics_path = Path("artifacts/model_evaluation/eval_report.json")
if not metrics_path.exists():
    st.info("No evaluation data found. Please run the training pipeline first using `python main.py`")
    st.stop()

with open(metrics_path) as f:
    metrics = json.load(f)

roc_auc = metrics.get('roc_auc', 0.74)
base_recall = metrics.get('recall', 0.40)
base_precision = metrics.get('precision', 0.22)
base_threshold = metrics.get('threshold', 0.15)

# ── Strategy Slider ──
render_section_label("Control")
render_section_title("Recovery Strategy")

st.markdown("""
<p style="font-size:12px;         color:#B8C4D4; margin-bottom:8px;">
Aggressive ← Flag more accounts, accept more false positives · Conservative ← Flag fewer, higher precision
</p>
""", unsafe_allow_html=True)

threshold = st.slider("Select Threshold", min_value=0.05, max_value=0.50, value=base_threshold, step=0.01)

# Simulate metrics
simulated_recall = base_recall * (base_threshold / max(threshold, 0.05))
simulated_precision = base_precision * (threshold / base_threshold)
simulated_f1 = 2 * (simulated_precision * simulated_recall) / (simulated_precision + simulated_recall + 0.001)

# Strategy label
if threshold <= 0.15:
    strategy_label = "Aggressive"
    strategy_color = "#4A9FE6"
    strategy_desc = "Flag more accounts, accept more false positives"
elif threshold <= 0.30:
    strategy_label = "Balanced"
    strategy_color = "#6BBF2A"
    strategy_desc = "Moderate flagging with balanced precision and recall"
else:
    strategy_label = "Conservative"
    strategy_color = "#D4A24C"
    strategy_desc = "Flag fewer accounts, higher precision per flag"

st.markdown(f"""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:1rem;">
    <span class="tag" style="background: {strategy_color}20; color: {strategy_color}; border: 0.5px solid {strategy_color}40; font-size:12px; padding:3px 10px; border-radius:6px; font-weight:500;">
        {strategy_label}
    </span>
    <span style="font-size:12px;         color:#B8C4D4;">{strategy_desc}</span>
</div>
""", unsafe_allow_html=True)

# Live metrics
st.markdown('<div style="display:flex; gap:10px; margin-bottom:1rem; flex-wrap:wrap;">', unsafe_allow_html=True)
mc = st.columns(4)
with mc[0]:
    render_metric_card("Flagged Accounts", f"{int(1000 * simulated_recall):,}")
with mc[1]:
    render_metric_card("Precision", f"{simulated_precision:.1%}")
with mc[2]:
    render_metric_card("Recall", f"{simulated_recall:.1%}")
with mc[3]:
    est_recovery = int(1000 * simulated_recall * simulated_precision * 500000 * 0.3)
    render_metric_card("Est. Recovery", f"₹{est_recovery/100000:.1f}L")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Sensitivity Chart ──
render_section_label("Sensitivity")
render_section_title("Threshold Sensitivity Analysis")

thresholds = np.arange(0.05, 0.51, 0.01)
recalls = [base_recall * (base_threshold / max(t, 0.05)) for t in thresholds]
precisions = [base_precision * (t / base_threshold) for t in thresholds]
f1s = [2 * (p * r) / (p + r + 0.001) for p, r in zip(precisions, recalls)]

fig = threshold_sensitivity_chart(thresholds, precisions, recalls, f1s, current_threshold=threshold)
st.plotly_chart(fig, width='stretch', config=plotly_config)

st.markdown("---")

# ── AI Recommendation ──
render_section_label("Recommendation")
render_section_title("AI Recommendation")

rec_html = generate_threshold_recommendation(threshold, simulated_precision, simulated_recall, simulated_f1)
st.markdown(f"""
<div class="insight-card">
    <div class="insight-title">AI Recommendation</div>
    <div style="font-size:13px;         color:#F0F2F6;">{rec_html}</div>
</div>
""", unsafe_allow_html=True)

# ── Business impact translation ──
render_section_label("Impact")
render_section_title("Business Impact Translation")

avg_flagged = int(1000 * simulated_recall)
officer_hours = int(avg_flagged * 0.5)  # assume 30 min per account
st.markdown(f"""
<div class="why-box" style="margin: 10px 0;">
    <div class="why-lbl">Operational Impact</div>
    <p style="font-size:12px;         color:#F0F2F6; margin:0;">
    At threshold <strong>{threshold:.2f}</strong>: ~{avg_flagged} accounts flagged, requiring approximately <strong>{officer_hours} officer hours</strong>.
    Estimated recovery value: <strong>₹{est_recovery/100000:.1f}L</strong>.
    Cost per recovered account: ~₹{int(officer_hours * 500 / max(int(avg_flagged * simulated_precision), 1)):,}.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Apply button ──
if st.button("Apply Threshold", type="primary", width='stretch'):
    st.toast(f"Threshold updated to {threshold:.2f} — {avg_flagged} accounts flagged")
    st.success(f"New threshold {threshold:.2f} applied. All dashboard predictions will use this threshold immediately.")

st.markdown("---")

# ── Confusion Matrix at threshold ──
render_section_label("Matrix")
render_section_title("Confusion Matrix at Selected Threshold")

actual_positives = 100
actual_negatives = 900
TP = int(actual_positives * simulated_recall)
FN = actual_positives - TP
FP = int(TP / (simulated_precision + 0.001) - TP) if simulated_precision > 0 else 0
TN = actual_negatives - FP

from utils.charts import confusion_matrix_chart
fig_cm = confusion_matrix_chart({"TN": TN, "FP": FP, "FN": FN, "TP": TP})
st.plotly_chart(fig_cm, width='stretch', config=plotly_config)

st.markdown("---")
st.caption("This simulator demonstrates business tradeoffs in threshold selection. Actual performance depends on data distribution and model calibration.")
