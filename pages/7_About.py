import streamlit as st
from utils.styles import inject_css, set_page_config
from utils.components import render_section_label, render_section_title

set_page_config("About", icon="ℹ️")
inject_css()

# ── Page-level styles ──
st.markdown("""
<style>
.about-hero-title {
    font-size: 28px;
    font-weight: 600;
    color: #F0F2F6;
    margin-bottom: 4px;
    letter-spacing: -0.3px;
}
.about-hero-sub {
    font-size: 14px;
    color: #B8C4D4;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}
.about-section-card {
    background: #141D2B;
    border: 0.5px solid #2A3B52;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.about-section-card:hover {
    border-color: #378ADD;
}
.about-section-card .asc-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4A9FE6;
    margin-bottom: 6px;
}
.about-section-card .asc-title {
    font-size: 16px;
    font-weight: 500;
    color: #FFFFFF;
    margin-bottom: 8px;
}
.about-section-card p {
    font-size: 13px;
    color: #B8C4D4;
    line-height: 1.7;
    margin: 0;
}
.goal-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 8px;
}
.goal-icon {
    font-size: 16px;
    flex-shrink: 0;
    margin-top: 1px;
}
.goal-text {
    font-size: 13px;
    color: #B8C4D4;
    line-height: 1.6;
}
.goal-text strong {
    color: #F0F2F6;
}
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 14px;
}
.step-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: rgba(74, 159, 230, 0.15);
    border: 0.5px solid rgba(74, 159, 230, 0.4);
    color: #4A9FE6;
    font-size: 12px;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.step-body .step-title {
    font-size: 13px;
    font-weight: 500;
    color: #F0F2F6;
    margin-bottom: 2px;
}
.step-body .step-desc {
    font-size: 12px;
    color: #B8C4D4;
    line-height: 1.55;
    margin: 0;
}
.tech-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 0.5px solid #2A3B52;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    color: #F0F2F6;
    margin: 4px 4px 4px 0;
}
.tech-pill .tp-icon { font-size: 16px; }
.tech-pill .tp-sub { font-size: 10px; color: #8B9BB4; display: block; }
.author-card {
    background: linear-gradient(135deg, rgba(74,159,230,0.08) 0%, rgba(107,191,42,0.06) 100%);
    border: 0.5px solid #2A3B52;
    border-radius: 14px;
    padding: 22px 24px;
    display: flex;
    align-items: center;
    gap: 18px;
    margin-top: 4px;
}
.author-avatar {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: linear-gradient(135deg, #185FA5, #3B6D11);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
    flex-shrink: 0;
}
.author-name {
    font-size: 16px;
    font-weight: 600;
    color: #F0F2F6;
    margin-bottom: 3px;
}
.author-role {
    font-size: 12px;
    color: #B8C4D4;
}
.disclaimer-box {
    background: rgba(212, 162, 76, 0.07);
    border: 0.5px solid rgba(212, 162, 76, 0.25);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 12px;
    color: #B8C4D4;
    line-height: 1.6;
    margin-top: 8px;
}
.disclaimer-box strong { color: #D4A24C; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──
st.markdown('<div class="about-hero-title"> AI-Based Loan Recovery Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="about-hero-sub">'
    'An AI-powered loan recovery prediction system — built to help financial institutions '
    'make faster, smarter, and more transparent recovery decisions.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ── What this project is ──
render_section_label("Overview")
render_section_title("What this project is")

st.markdown("""
<div class="about-section-card">
    <div class="asc-label">Project Summary</div>
    <p>
    <strong style="color:#F0F2F6;">Recovery Intelligence</strong> is an end-to-end machine learning
    system that predicts the probability of successful loan recovery for delinquent or at-risk borrowers.
    It combines advanced feature engineering, gradient-boosted classification models, and SHAP-based
    explainability into a unified, interactive decision-support dashboard.<br><br>
    The system ingests borrower financial data, bureau credit history, repayment patterns, and social
    risk indicators — and produces a recovery probability score, risk tier classification, and a
    recommended next action for every borrower.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Problem Statement ──
render_section_label("Problem")
render_section_title("Problem Statement")

prob_cols = st.columns(2)
with prob_cols[0]:
    st.markdown("""
    <div class="about-section-card" style="height:100%;">
        <div class="asc-label">The Challenge</div>
        <p>
        Financial institutions manage thousands of delinquent loans with limited recovery teams.
        Traditional collection strategies rely on manual review and generic rules — leading to missed
        recoveries, wasted effort on low-probability cases, and inconsistent decision-making.
        Without knowing <em style="color:#5CB0FF; font-style:normal;">which loans are actually likely to recover</em>,
        collection teams work blind and resources are misallocated.
        </p>
    </div>
    """, unsafe_allow_html=True)

with prob_cols[1]:
    st.markdown("""
    <div class="about-section-card" style="height:100%;">
        <div class="asc-label">Business Impact</div>
        <p>
        <strong style="color:#E24B4A;">Revenue loss</strong> — Recoverable loans slip through due to poor prioritization.<br><br>
        <strong style="color:#D4A24C;">Operational cost</strong> — Officers waste hours on borrowers who will never pay.<br><br>
        <strong style="color:#4A9FE6;">Regulatory risk</strong> — Opaque decisions without audit trails or explanations.<br><br>
        <strong style="color:#B8C4D4;">Slow response</strong> — Manual analysis takes hours; recovery windows close fast.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Goals ──
render_section_label("Goals")
render_section_title("What this project fulfills")

goal_cols = st.columns(2)
goals_left = [
    ("🎯", "Predict Recovery Probability", "Assign each borrower a machine-learning-derived score indicating their likelihood of loan recovery."),
    ("📊", "Portfolio Prioritization", "Rank and tier the entire portfolio by risk so collection teams always know where to focus first."),
    ("🔍", "Explainable AI Decisions", "Use SHAP to explain every prediction in plain language — building trust and enabling audit trails."),
]
goals_right = [
    ("⚙️", "Strategy Optimization", "Simulate business-risk tradeoffs by adjusting recovery thresholds and measuring monetary impact."),
    ("💰", "Reduce Default Losses", "Improve recovery rates by surfacing high-probability accounts that would otherwise be missed."),
    ("⚡", "Real-time Decision Support", "Process single borrowers or entire portfolios in seconds — not hours of manual review."),
]

with goal_cols[0]:
    for icon, title, desc in goals_left:
        st.markdown(f"""
        <div class="about-section-card">
            <div class="goal-row">
                <div class="goal-icon">{icon}</div>
                <div>
                    <div style="font-size:13px; font-weight:500; color:#F0F2F6; margin-bottom:4px;">{title}</div>
                    <div class="goal-text">{desc}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with goal_cols[1]:
    for icon, title, desc in goals_right:
        st.markdown(f"""
        <div class="about-section-card">
            <div class="goal-row">
                <div class="goal-icon">{icon}</div>
                <div>
                    <div style="font-size:13px; font-weight:500; color:#F0F2F6; margin-bottom:4px;">{title}</div>
                    <div class="goal-text">{desc}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Dataset ──
render_section_label("Data")
render_section_title("Dataset Used")

st.markdown("""
<div class="about-section-card">
    <div class="asc-label">Home Credit Risk Dataset</div>
    <p>
    The project is built on the <strong style="color:#F0F2F6;">Home Credit Default Risk</strong> dataset — a real-world,
    large-scale financial dataset containing:
    </p>
    <br>
""", unsafe_allow_html=True)

data_cols = st.columns(5)
dataset_items = [
    ("🧾", "Application Data", "Borrower financials & loan details"),
    ("🏦", "Bureau History", "External credit bureau records"),
    ("🔄", "Repayment Behavior", "Historical payment patterns"),
    ("👥", "Social Risk Indicators", "Household & social defaults"),
    ("📈", "External Scores", "EXT_SOURCE credit risk scores"),
]
for col, (icon, title, desc) in zip(data_cols, dataset_items):
    with col:
        st.markdown(f"""
        <div style="text-align:center; padding:12px 8px;">
            <div style="font-size:24px; margin-bottom:6px;">{icon}</div>
            <div style="font-size:12px; font-weight:500; color:#F0F2F6; margin-bottom:3px;">{title}</div>
            <div style="font-size:11px; color:#8B9BB4;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ── Project Workflow ──
render_section_label("Workflow")
render_section_title("How it was built")

wf_cols = st.columns(2)
steps_left = [
    ("1", "Data Collection & Cleaning", "Borrower application + bureau datasets. Missing value handling, anomaly detection, outlier treatment, and duplicate checks."),
    ("2", "Feature Engineering", "Financial ratios, bureau behavioral features, employment indicators, social risk features, and external score aggregation."),
    ("3", "Encoding & Transformation", "Binary encoding, one-hot encoding, and frequency encoding for categorical features."),
    ("4", "Model Building", "LightGBM & XGBoost gradient-boosted classifiers optimised for ROC-AUC on the imbalanced recovery target."),
]
steps_right = [
    ("5", "Model Optimization", "Threshold tuning, precision-recall tradeoff analysis, and strategy simulation at business level."),
    ("6", "Explainable AI", "SHAP summary plots, waterfall breakdowns, and per-borrower factor analysis for regulatory transparency."),
    ("7", "Evaluation", "ROC-AUC, Recall, Precision, F1, and Confusion Matrix benchmarked against baseline models."),
    ("8", "Deployment", "Streamlit multi-page dashboard with batch processing, single-borrower assessment, and live model health tracking."),
]

with wf_cols[0]:
    for num, title, desc in steps_left:
        st.markdown(f"""
        <div class="step-row">
            <div class="step-num">{num}</div>
            <div class="step-body">
                <div class="step-title">{title}</div>
                <p class="step-desc">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

with wf_cols[1]:
    for num, title, desc in steps_right:
        st.markdown(f"""
        <div class="step-row">
            <div class="step-num">{num}</div>
            <div class="step-body">
                <div class="step-title">{title}</div>
                <p class="step-desc">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Tech Stack ──
render_section_label("Stack")
render_section_title("Technology & Architecture")

tech_items = [
    ("🐍", "Python", "Core ML pipeline"),
    ("⚡", "LightGBM", "Gradient boosting"),
    ("🌲", "XGBoost", "Ensemble model"),
    ("🔬", "SHAP", "Explainability"),
    ("🧪", "Scikit-learn", "Preprocessing"),
    ("🐼", "Pandas / NumPy", "Data manipulation"),
    ("📊", "Plotly", "Interactive charts"),
    ("🎯", "Streamlit", "Web dashboard"),
]

tech_html = '<div style="display:flex; flex-wrap:wrap; gap:0;">'
for icon, name, role in tech_items:
    tech_html += f"""
    <div class="tech-pill">
        <span class="tp-icon">{icon}</span>
        <span>
            <span style="display:block; font-weight:500;">{name}</span>
            <span class="tp-sub">{role}</span>
        </span>
    </div>"""
tech_html += "</div>"
st.markdown(tech_html, unsafe_allow_html=True)

st.markdown("---")

# ── Author ──
render_section_label("Author")
render_section_title("Built by")

st.markdown("""
<div class="author-card">
    <div class="author-avatar">A</div>
    <div>
        <div class="author-name">Ankit</div>
        <div class="author-role">ML Engineer · AI-Based Loan Recovery Prediction System</div>
        <div style="margin-top:8px; display:flex; gap:6px; flex-wrap:wrap;">
            <span style="background:rgba(74,159,230,0.1); border:0.5px solid rgba(74,159,230,0.3); border-radius:6px; padding:2px 10px; font-size:11px; color:#4A9FE6;">Machine Learning</span>
            <span style="background:rgba(107,191,42,0.1); border:0.5px solid rgba(107,191,42,0.3); border-radius:6px; padding:2px 10px; font-size:11px; color:#6BBF2A;">Explainable AI</span>
            <span style="background:rgba(212,162,76,0.1); border:0.5px solid rgba(212,162,76,0.3); border-radius:6px; padding:2px 10px; font-size:11px; color:#D4A24C;">Financial Analytics</span>
            <span style="background:rgba(226,75,74,0.1); border:0.5px solid rgba(226,75,74,0.3); border-radius:6px; padding:2px 10px; font-size:11px; color:#E24B4A;">Credit Risk</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Disclaimer ──
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Disclaimer:</strong> This application provides AI-driven insights for decision support only.
    All predictions are probabilistic estimates based on historical data patterns. Final loan recovery
    decisions must include human business review and comply with applicable regulatory requirements.
    Model outputs should not be used as the sole basis for any financial or legal action.
</div>
""", unsafe_allow_html=True)

st.caption("Recovery Intelligence · AI-Based Loan Recovery Prediction System · Built with LightGBM, SHAP & Streamlit")
