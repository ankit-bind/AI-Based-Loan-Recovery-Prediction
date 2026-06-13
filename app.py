import streamlit as st
from pathlib import Path
from utils.styles import inject_css, set_page_config
from utils.components import render_section_title, render_section_label

set_page_config("Home", icon="", layout="wide")
inject_css()

st.markdown("""
<style>
.landing-title {
    font-size: 28px;
    font-weight: 500;
    color: #F0F2F6;
    margin-bottom: 6px;
}
.landing-sub {
    font-size: 14px;
    color: #B8C4D4;
    margin-bottom: 1.5rem;
}
.landing-card {
    background: #141D2B;
    border: 0.5px solid #2A3B52;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s;
}
.landing-card:hover {
    border-color: #378ADD;
}
.landing-card .card-title {
    font-size: 14px;
    font-weight: 500;
    color: #F0F2F6;
    margin-bottom: 4px;
}
.landing-card .card-desc {
    font-size: 12px;
    color: #B8C4D4;
    margin: 0;
}
.landing-card .card-icon {
    font-size: 20px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="landing-title">AI-Based Loan Recovery Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="landing-sub">AI-powered loan recovery probability prediction for enterprise decision support</div>', unsafe_allow_html=True)

# Quick navigation cards
render_section_label("Navigate")

pages = [
    ("", "Portfolio", "Executive banking overview and portfolio health", "1_Portfolio"),
    ("", "Single Borrower", "Predict recovery probability for an individual borrower", "2_Single_Borrower"),
    ("", "Portfolio Analysis", "Process multiple loan applications via CSV upload", "3_Portfolio_Analysis"),
    ("", "Model Health", "View model evaluation metrics and visualizations", "4_Model_Health"),
    ("", "Decision Intelligence", "Understand model predictions with explainable AI", "5_Decision_Intelligence"),
    ("", "Strategy Config", "Interactive business-risk tradeoff analysis", "6_Strategy_Config"),
    ("ℹ️", "About", "Project overview, goals, dataset & tech stack", "7_About"),
]

cols = st.columns(3)
for i, (icon, title, desc, page) in enumerate(pages):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="landing-card">
            <div class="card-icon">{icon}</div>
            <div class="card-title">{title}</div>
            <p class="card-desc">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# About / product section
render_section_label("About")
render_section_title("What this platform does")

st.markdown("""
<p style="font-size:14px;     color:#F0F2F6; line-height:1.65;">
<b>Recovery Intelligence</b> is an AI-powered decision-support system that predicts the probability of loan recovery
for delinquent or at-risk borrowers. It helps banks and financial institutions reduce default losses,
optimize collection efforts, and make data-driven recovery decisions — all in one unified dashboard.
</p>
""", unsafe_allow_html=True)

# ── The Problem ──
render_section_label("Problem")
render_section_title("Why this matters")

problem_cols = st.columns(2)
with problem_cols[0]:
    st.markdown("""
    <div class="why-box" style="margin-bottom:10px;">
        <div class="why-lbl">The challenge</div>
        <p style="font-size:12px;     color:#F0F2F6; margin:0; line-height:1.65;">
        Banks manage thousands of delinquent loans with limited recovery staff. Traditional approaches rely on
        manual review and generic rules, which leads to missed recoveries, wasted effort on low-probability cases,
        and inconsistent decision-making across teams. Without knowing which loans are likely to recover,
        collection teams work blind.
        </p>
    </div>
    """, unsafe_allow_html=True)

with problem_cols[1]:
    st.markdown("""
    <div class="why-box" style="margin-bottom:10px;">
        <div class="why-lbl">The impact</div>
        <p style="font-size:12px;     color:#F0F2F6; margin:0; line-height:1.65;">
        <strong>Revenue loss</strong> — Recoverable loans slip through cracks due to poor prioritization.<br>
        <strong>Operational cost</strong> — Officers spend time on borrowers who will never pay.<br>
        <strong>Regulatory risk</strong> — Opaque decisions without audit trails or explanations.<br>
        <strong>Slow response</strong> — Manual analysis takes hours; recovery windows close fast.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── The Solution ──
render_section_label("Solution")
render_section_title("How this platform resolves it")

st.markdown("""
<div class="why-box" style="margin:10px 0;">
    <div class="why-lbl">Recovery Intelligence in action</div>
    <p style="font-size:12px;     color:#F0F2F6; margin:0; line-height:1.65;">
    1. <strong>Predict</strong> — Machine learning models analyze borrower data and assign a recovery probability score.<br>
    2. <strong>Prioritize</strong> — Risk tiers (Critical, Low, Moderate, High) sort the portfolio so teams focus on what matters.<br>
    3. <strong>Explain</strong> — SHAP-based explainability shows exactly which factors drove the prediction, in plain language.<br>
    4. <strong>Act</strong> — Every prediction comes with a recommended action: escalate, call, review, or continue servicing.<br>
    5. <strong>Monitor</strong> — Live model health tracking ensures predictions stay accurate and drift is caught early.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Key Capabilities ──
render_section_label("Capabilities")
render_section_title("What you can do here")

cap_cols = st.columns(3)
with cap_cols[0]:
    st.markdown("""
    <div class="why-box" style="margin-bottom:10px;">
        <div class="why-lbl">Portfolio Dashboard</div>
        <p style="font-size:12px;     color:#F0F2F6; margin:0; line-height:1.65;">
        View real-time portfolio health: total borrowers, recovery rate, default rate, critical accounts,
        and AI-generated insights. Top critical accounts are surfaced automatically with one-click actions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="why-box" style="margin-bottom:10px;">
        <div class="why-lbl">Single Borrower</div>
        <p style="font-size:12px;     color:#F0F2F6; margin:0; line-height:1.65;">
        Enter a borrower's profile and get an instant recovery probability assessment with a borrower intelligence card
        showing risk tier, top SHAP factors, confidence level, and a recommended next step.
        </p>
    </div>
    """, unsafe_allow_html=True)

with cap_cols[1]:
    st.markdown("""
    <div class="why-box" style="margin-bottom:10px;">
        <div class="why-lbl">Portfolio Analysis</div>
        <p style="font-size:12px;     color:#F0F2F6; margin:0; line-height:1.65;">
        Upload a CSV of loan accounts and process up to 10,000 borrowers in a single batch.
        Results are risk-sorted with critical accounts flagged first, and downloadable reports are generated.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="why-box" style="margin-bottom:10px;">
        <div class="why-lbl">Model Health</div>
        <p style="font-size:12px;     color:#F0F2F6; margin:0; line-height:1.65;">
        Track model performance with ROC-AUC, precision, recall, and F1 metrics against benchmarks.
        Compare LightGBM vs XGBoost and review confusion matrices with False Negative highlighting.
        </p>
    </div>
    """, unsafe_allow_html=True)

with cap_cols[2]:
    st.markdown("""
    <div class="why-box" style="margin-bottom:10px;">
        <div class="why-lbl">Decision Intelligence</div>
        <p style="font-size:12px;     color:#F0F2F6; margin:0; line-height:1.65;">
        Understand model predictions through SHAP explainability: summary plots, bar charts, waterfall breakdowns,
        and regulatory-grade audit logs. Build trust in every automated decision.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="why-box" style="margin-bottom:10px;">
        <div class="why-lbl">Strategy Config</div>
        <p style="font-size:12px;     color:#F0F2F6; margin:0; line-height:1.65;">
        Adjust the recovery threshold with a live business simulator. See how Aggressive, Balanced, or Conservative
        strategies affect flagged accounts, precision, recall, and estimated monetary recovery.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Who it is for ──
render_section_label("Audience")
render_section_title("Who this is built for")

audience_cols = st.columns(4)
with audience_cols[0]:
    st.markdown("""
    <div class="why-box" style="text-align:center;">
        <div style="font-size:20px; margin-bottom:6px;"></div>
        <p style="font-size:13px; font-weight:500; color:#F0F2F6; margin-bottom:3px;">Recovery Officers</p>
        <p style="font-size:11px;     color:#B8C4D4; margin:0;">Single-borrower decisions in &lt;30 sec</p>
    </div>
    """, unsafe_allow_html=True)
with audience_cols[1]:
    st.markdown("""
    <div class="why-box" style="text-align:center;">
        <div style="font-size:20px; margin-bottom:6px;"></div>
        <p style="font-size:13px; font-weight:500; color:#F0F2F6; margin-bottom:3px;">Credit Risk Analysts</p>
        <p style="font-size:11px;     color:#B8C4D4; margin:0;">Model validation &amp; SHAP deep-dives</p>
    </div>
    """, unsafe_allow_html=True)
with audience_cols[2]:
    st.markdown("""
    <div class="why-box" style="text-align:center;">
        <div style="font-size:20px; margin-bottom:6px;"></div>
        <p style="font-size:13px; font-weight:500; color:#F0F2F6; margin-bottom:3px;">Banking Managers</p>
        <p style="font-size:11px;     color:#B8C4D4; margin:0;">Portfolio health in 2–3 minutes</p>
    </div>
    """, unsafe_allow_html=True)
with audience_cols[3]:
    st.markdown("""
    <div class="why-box" style="text-align:center;">
        <div style="font-size:20px; margin-bottom:6px;"></div>
        <p style="font-size:13px; font-weight:500; color:#F0F2F6; margin-bottom:3px;">Collection Teams</p>
        <p style="font-size:11px;     color:#B8C4D4; margin:0;">Batch-prioritized daily workqueues</p>
    </div>
    """, unsafe_allow_html=True)

# ── Technology ──
render_section_label("Stack")
render_section_title("Technology & Architecture")

st.markdown("""
<p style="font-size:12px;     color:#B8C4D4; margin-bottom:10px;">
Built with open-source, enterprise-grade tools for reliability, transparency, and speed.
</p>
""", unsafe_allow_html=True)

stack_cols = st.columns(4)
with stack_cols[0]:
    st.markdown("""
    <div class="why-box" style="text-align:center;">
        <div style="font-size:20px; margin-bottom:6px;"></div>
        <p style="font-size:13px; font-weight:500; color:#F0F2F6; margin-bottom:3px;">Python</p>
        <p style="font-size:11px;     color:#B8C4D4; margin:0;">Core ML pipeline</p>
    </div>
    """, unsafe_allow_html=True)
with stack_cols[1]:
    st.markdown("""
    <div class="why-box" style="text-align:center;">
        <div style="font-size:20px; margin-bottom:6px;"></div>
        <p style="font-size:13px; font-weight:500; color:#F0F2F6; margin-bottom:3px;">LightGBM</p>
        <p style="font-size:11px;     color:#B8C4D4; margin:0;">Gradient boosting</p>
    </div>
    """, unsafe_allow_html=True)
with stack_cols[2]:
    st.markdown("""
    <div class="why-box" style="text-align:center;">
        <div style="font-size:20px; margin-bottom:6px;"></div>
        <p style="font-size:13px; font-weight:500; color:#F0F2F6; margin-bottom:3px;">Plotly</p>
        <p style="font-size:11px;     color:#B8C4D4; margin:0;">Interactive charts</p>
    </div>
    """, unsafe_allow_html=True)
with stack_cols[3]:
    st.markdown("""
    <div class="why-box" style="text-align:center;">
        <div style="font-size:20px; margin-bottom:6px;"></div>
        <p style="font-size:13px; font-weight:500; color:#F0F2F6; margin-bottom:3px;">SHAP</p>
        <p style="font-size:11px;     color:#B8C4D4; margin:0;">Explainability</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("This application provides AI-driven insights for decision support. Final loan decisions should include business review and human verification.")
