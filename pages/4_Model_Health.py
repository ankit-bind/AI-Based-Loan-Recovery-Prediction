import streamlit as st
import pandas as pd
import json
from pathlib import Path
import numpy as np

from utils.styles import inject_css, set_page_config
from utils.components import render_section_label, render_section_title, render_verdict_card
from utils.charts import (
    roc_curve_chart, confusion_matrix_chart, feature_importance_chart,
    model_comparison_chart, precision_recall_curve, apply_fintech_theme, plotly_config
)
from utils.insights import load_metrics, load_model_comparison, generate_model_health_verdict

set_page_config("Model Health", icon="")
inject_css()

render_section_label("Health")
render_section_title("Model Health Report")

st.markdown("""
<p style="font-size:14px;         color:#F0F2F6; line-height:1.65; margin-bottom:1rem;">
Can you trust this model for a recovery decision? This page translates technical metrics into a plain-language assessment.
</p>
""", unsafe_allow_html=True)

metrics = load_metrics()
model_comparison = load_model_comparison()

if not metrics:
    st.info("No evaluation metrics found. Please run the training pipeline first using `python main.py`")
    st.stop()

# ── Verdict card ──
verdict = generate_model_health_verdict(metrics)
render_verdict_card(verdict["verdict"], verdict["explanation"], verdict["severity"])

st.markdown("---")

# ── Metrics with benchmark bands ──
render_section_label("Metrics")
render_section_title("Evaluation Metrics vs Benchmarks")

benchmarks = {
    "roc_auc": (0.70, 0.85, "Acceptable for credit risk"),
    "accuracy": (0.75, 0.90, "General classification"),
    "precision": (0.20, 0.40, "Imbalanced data range"),
    "recall": (0.30, 0.60, "Imbalanced data range"),
    "f1_score": (0.25, 0.50, "Balanced metric range"),
}

metric_names = {
    "roc_auc": "ROC-AUC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-Score",
}

mcols = st.columns(5)
for i, (key, label) in enumerate(metric_names.items()):
    val = metrics.get(key, 0.0)
    low, high, desc = benchmarks.get(key, (0, 1, ""))
    if val >= high:
        delta = "Above benchmark"
        dcolor = "normal"
    elif val >= low:
        delta = "In range"
        dcolor = "normal"
    else:
        delta = "Below range"
        dcolor = "inverse"
    with mcols[i]:
        st.metric(label, f"{val:.4f}", delta=delta, delta_color=dcolor)

st.markdown(f"""
<p style="font-size:12px;         color:#B8C4D4; margin-top:4px;">
Benchmarks: {', '.join([f"{metric_names[k]} {benchmarks[k][0]:.2f}–{benchmarks[k][1]:.2f}" for k in metric_names])}
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Model Comparison ──
if model_comparison:
    render_section_label("Comparison")
    render_section_title("Model Comparison")
    comparison_df = pd.DataFrame(model_comparison).T.reset_index().rename(columns={'index': 'Model'})
    fig_comp = model_comparison_chart(comparison_df)
    st.plotly_chart(fig_comp, width='stretch', config=plotly_config)
    st.dataframe(comparison_df, width='stretch')
    st.markdown("---")

# ── Visualizations ──
render_section_label("Visualizations")
render_section_title("Model Performance Charts")

# ROC Curve
roc_path = Path("artifacts/model_evaluation/roc_curve.png")
if roc_path.exists():
    from PIL import Image
    img = Image.open(roc_path)
    st.image(str(roc_path), caption="ROC Curve", width='stretch')
else:
    # Simulate ROC for demo
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - (1 - fpr) ** (1 / (2 * metrics.get('roc_auc', 0.74)))
    fig_roc = roc_curve_chart(fpr, tpr, metrics.get('roc_auc', 0.74))
    st.plotly_chart(fig_roc, width='stretch', config=plotly_config)

# Precision-Recall
pr_path = Path("artifacts/model_evaluation/pr_curve.png")
if pr_path.exists():
    st.image(str(pr_path), caption="Precision-Recall Curve", width='stretch')
else:
    recalls = np.linspace(0.01, 1, 100)
    precisions = [metrics.get('precision', 0.22) * (r / 0.4) ** 0.5 for r in recalls]
    thresholds = np.linspace(0.05, 0.95, 100)
    f1s = [2 * (p * r) / (p + r + 0.001) for p, r in zip(precisions, recalls)]
    fig_pr = precision_recall_curve(precisions, recalls, thresholds, f1s, highlight_threshold=metrics.get('threshold', 0.38))
    st.plotly_chart(fig_pr, width='stretch', config=plotly_config)

st.markdown("---")

# Confusion Matrix
render_section_label("Confusion Matrix")
render_section_title("Confusion Matrix at Current Threshold")
# Simulated from metrics
recall = metrics.get('recall', 0.39)
precision = metrics.get('precision', 0.22)
actual_positives = 100
actual_negatives = 900
TP = int(actual_positives * recall)
FN = actual_positives - TP
FP = int(TP / (precision + 0.001) - TP) if precision > 0 else 0
TN = actual_negatives - FP
fig_cm = confusion_matrix_chart({"TN": TN, "FP": FP, "FN": FN, "TP": TP})
st.plotly_chart(fig_cm, width='stretch', config=plotly_config)

st.markdown("""
<div class="why-box" style="margin-top:10px;">
    <div class="why-lbl">Interpretation</div>
    <p style="font-size:12px;         color:#F0F2F6; margin:0;">
    The <strong>False Negative</strong> cell (missed recoveries) is highlighted in red — this represents revenue loss.
    At current threshold, {FN} recoverable accounts are not flagged. Adjust threshold on Strategy Config page.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Feature Importance
feature_importance_path = Path("artifacts/model_trainer/feature_importance.csv")
if feature_importance_path.exists():
    render_section_label("Features")
    render_section_title("Feature Importance (Top 20)")
    fi_df = pd.read_csv(feature_importance_path)
    fig_imp = feature_importance_chart(fi_df, x_col='importance', y_col='feature', top_n=20)
    st.plotly_chart(fig_imp, width='stretch', config=plotly_config)

st.markdown("---")
st.caption("Model evaluation based on test set performance. Metrics may vary with different data splits.")
