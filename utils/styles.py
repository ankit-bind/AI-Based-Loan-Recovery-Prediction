import streamlit as st

# ── Global CSS Injection ──
def inject_css():
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        min-width: 240px !important;
        max-width: 240px !important;
    }
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-h4zwh2 {
        padding-top: 1rem;
    }

    /* ── Sidebar nav section labels ── */
    .sidebar-section {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8B9BB4;
        padding: 16px 10px 6px 18px;
        margin-top: 8px;
    }
    .sidebar-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 14px;
        margin: 2px 8px;
        border-radius: 8px;
        font-size: 13px;
        color: #B8C4D4;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .sidebar-item:hover {
        background: #1A2535;
        color: #F0F2F6;
    }
    .sidebar-item.active {
        background: #162033;
        color: #F0F2F6;
        font-weight: 500;
        border: 0.5px solid #2A3B52;
    }

    /* ── Pulsing live dot ── */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #3B6D11;
        display: inline-block;
        animation: pulse 2s infinite;
    }

    /* ── Metric card ── */
    .metric-card {
        background: #141D2B !important;
        border: 0.5px solid #2A3B52 !important;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-card .val {
        font-size: 22px;
        font-weight: 500;
        color: #F0F2F6 !important;
        font-variant-numeric: tabular-nums;
    }
    .metric-card .lbl {
        font-size: 11px;
        color: #B8C4D4 !important;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-card .delta {
        font-size: 12px;
        margin-top: 4px;
    }

    /* ── Risk badge ── */
    .risk-badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 6px;
    }
    .badge-critical {
        background: rgba(226, 75, 74, 0.12);
        color: #E24B4A;
        border: 0.5px solid rgba(226, 75, 74, 0.3);
    }
    .badge-low {
        background: rgba(186, 117, 23, 0.12);
        color: #D4A24C;
        border: 0.5px solid rgba(186, 117, 23, 0.3);
    }
    .badge-moderate {
        background: rgba(24, 95, 165, 0.12);
        color: #4A9FE6;
        border: 0.5px solid rgba(24, 95, 165, 0.3);
    }
    .badge-high {
        background: rgba(59, 109, 17, 0.12);
        color: #6BBF2A;
        border: 0.5px solid rgba(59, 109, 17, 0.3);
    }

    /* ── Borrower card ── */
    .borrower-card {
        background: #141D2B;
        border: 0.5px solid #2A3B52;
        border-radius: 14px;
        padding: 20px;
        max-width: 680px;
        margin: 12px 0;
    }
    .borrower-card .header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 14px;
    }
    .borrower-card .avatar {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: rgba(226, 75, 74, 0.15);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 15px;
        color: #E24B4A;
        flex-shrink: 0;
    }
    .borrower-card .meta {
        flex: 1;
    }
    .borrower-card .meta .title {
        font-weight: 500;
        font-size: 15px;
        color: #F0F2F6;
        margin: 0;
    }
    .borrower-card .meta .subtitle {
        font-size: 12px;
        color: #B8C4D4;
        margin: 2px 0 0;
    }
    .borrower-card .status {
        text-align: right;
    }
    .borrower-card .status .prob {
        font-size: 11px;
        color: #B8C4D4;
        margin: 4px 0 0;
    }
    .borrower-card .shap-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 12px;
        color: #F0F2F6;
    }
    .borrower-card .shap-row .bar-bg {
        width: 80px;
        height: 6px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        position: relative;
        flex-shrink: 0;
    }
    .borrower-card .shap-row .bar-fill {
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        border-radius: 3px;
    }
    .borrower-card .action-box {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 12px;
        margin-top: 10px;
    }
    .borrower-card .action-box .label {
        color: #B8C4D4;
    }
    .borrower-card .action-box .action {
        color: #E24B4A;
        font-weight: 500;
    }

    /* ── Standalone SHAP row (used outside borrower-card) ── */
    .shap-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
        font-size: 12px;
        color: #F0F2F6;
    }
    .shap-row .bar-bg {
        width: 80px;
        height: 6px;
        background: rgba(255,255,255,0.06);
        border-radius: 3px;
        position: relative;
        flex-shrink: 0;
    }
    .shap-row .bar-fill {
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        border-radius: 3px;
    }

    /* ── Banner ── */
    .health-banner {
        background: rgba(24, 95, 165, 0.12);
        border: 0.5px solid rgba(24, 95, 165, 0.3);
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 13px;
        color: #F0F2F6;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .health-banner strong {
        color: #FFFFFF;
    }

    /* ── Insight card ── */
    .insight-card {
        background: rgba(255,255,255,0.04);
        border: 0.5px solid #2A3B52;
        border-radius: 12px;
        padding: 14px 16px;
        font-size: 13px;
        color: #F0F2F6;
        margin: 10px 0;
    }
    .insight-card .insight-title {
        font-weight: 500;
        color: #FFFFFF;
        margin-bottom: 6px;
        font-size: 13px;
    }
    .insight-card em {
        color: #5CB0FF;
        font-style: normal;
        font-weight: 500;
    }

    /* ── Divider ── */
    .bp-divider {
        height: 0.5px;
        background: #2A3B52;
        margin: 12px 0;
    }

    /* ── Why box ── */
    .why-box {
        background: #141D2B;
        border-radius: 10px;
        padding: 12px 14px;
        margin: 6px 0;
    }
    .why-box .why-lbl {
        font-size: 10px;
        font-weight: 600;
        color: #B8C4D4;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    /* ── Priority badge ── */
    .priority-badge {
        display: inline-block;
        font-size: 10px;
        padding: 2px 7px;
        border-radius: 4px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .p1 { background: rgba(226, 75, 74, 0.15); color: #E24B4A; }
    .p2 { background: rgba(186, 117, 23, 0.15); color: #D4A24C; }
    .p3 { background: rgba(24, 95, 165, 0.15); color: #4A9FE6; }
    .p4 { background: rgba(255,255,255,0.05); color: #B8C4D4; border: 0.5px solid #2A3B52; }

    /* ── Pipeline stage ── */
    .pipeline-stage {
        background: #141D2B;
        border: 0.5px solid #2A3B52;
        border-radius: 12px;
        padding: 14px;
        margin: 6px 0;
    }
    .pipeline-stage .stage-title {
        font-size: 12px;
        font-weight: 500;
        color: #F0F2F6;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .pipeline-stage .stage-desc {
        font-size: 12px;
        color: #B8C4D4;
        margin: 0;
    }

    /* ── Table spec ── */
    table.spec {
        width: 100%;
        border-collapse: collapse;
        font-size: 12.5px;
        margin: 8px 0;
    }
    table.spec th {
        text-align: left;
        color: #B8C4D4;
        font-weight: 500;
        font-size: 11px;
        padding: 6px 8px;
        border-bottom: 0.5px solid #2A3B52;
    }
    table.spec td {
        padding: 7px 8px;
        color: #F0F2F6;
        border-bottom: 0.5px solid #2A3B52;
        vertical-align: top;
    }
    table.spec td:first-child {
        color: #F0F2F6;
        font-weight: 500;
        white-space: nowrap;
    }

    /* ── Arch box ── */
    .arch-box {
        border: 0.5px solid #2A3B52;
        border-radius: 12px;
        padding: 12px;
        margin: 6px 0;
    }
    .arch-box .arch-title {
        font-size: 12px;
        font-weight: 500;
        color: #F0F2F6;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* ── Wireframe ── */
    .wireframe {
        background: #141D2B;
        border: 0.5px solid #2A3B52;
        border-radius: 12px;
        padding: 14px;
        margin: 10px 0;
        font-size: 12px;
        color: #B8C4D4;
    }

    /* ── Empty state ── */
    .empty-state {
        background: rgba(255,255,255,0.02);
        border: 1px dashed #2A3B52;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        [data-testid="column"] {
            flex: 0 0 100% !important;
        }
        .metric-card .val {
            font-size: 18px !important;
        }
    }

    /* ── Hide Streamlit default header padding ── */
    .block-container {
        padding-top: 1rem !important;
    }

    /* ── Tag ── */
    .tag {
        display: inline-block;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 6px;
        margin: 2px 2px 2px 0;
    }
    .tag-blue {
        background: rgba(55, 138, 221, 0.12);
        color: #4A9FE6;
    }
    .tag-green {
        background: rgba(59, 109, 17, 0.12);
        color: #6BBF2A;
    }
    .tag-amber {
        background: rgba(186, 117, 23, 0.12);
        color: #D4A24C;
    }
    .tag-red {
        background: rgba(226, 75, 74, 0.12);
        color: #E24B4A;
    }
    .tag-gray {
        background: rgba(255,255,255,0.04);
        color: #B8C4D4;
        border: 0.5px solid #2A3B52;
    }

    /* ── Sticky top bar ── */
    .sticky-bar {
        position: sticky;
        top: 0;
        z-index: 100;
        background: rgba(15, 24, 35, 0.92);
        backdrop-filter: blur(12px);
        border-bottom: 0.5px solid #2A3B52;
        padding: 6px 16px;
        font-size: 11px;
        color: #B8C4D4;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
    }
    .sticky-bar .pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        border: 0.5px solid #2A3B52;
        border-radius: 99px;
        padding: 2px 8px;
        font-size: 11px;
        color: #F0F2F6;
    }

    /* ── Streamlit expander cursor fix ── */
    [data-testid="stExpander"] summary {
        cursor: pointer !important;
        caret-color: transparent !important;
        user-select: none !important;
        -webkit-user-select: none !important;
    }
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"] summary:active {
        outline: none !important;
        box-shadow: none !important;
    }

    /* ── Streamlit native elements dark mode fixes ── */
    [data-testid="stMetric"] {
        background: #141D2B !important;
        border: 0.5px solid #2A3B52 !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }
    [data-testid="stMetric"] > div {
        color: #F0F2F6 !important;
    }
    [data-testid="stMetric"] label {
        color: #B8C4D4 !important;
    }
    .stDataFrame, [data-testid="stDataFrame"] {
        background: #141D2B !important;
    }
    [data-testid="stDataFrame"] th {
        color: #F0F2F6 !important;
        background: #1A2535 !important;
    }
    [data-testid="stDataFrame"] td {
        color: #F0F2F6 !important;
        background: #141D2B !important;
    }
    [data-testid="stAlert"], [data-testid="stNotification"] {
        color: #F0F2F6 !important;
    }
    .stMarkdown, [data-testid="stMarkdownContainer"] {
        color: #F0F2F6 !important;
    }
    /* Ensure all Streamlit containers use dark background */
    [data-testid="stVerticalBlock"] > div,
    [data-testid="stHorizontalBlock"] > div,
    [data-testid="column"] > div {
        background: transparent !important;
    }
    /* Ensure custom HTML containers don't show white background */
    .element-container .stMarkdown {
        background: transparent !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ── Page Config ──
def set_page_config(title: str, icon: str = "", layout: str = "wide"):
    st.set_page_config(
        page_title=f"Recovery Intelligence — {title}",
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded"
    )


# ── Sticky Model Status Bar ──
def model_status_bar():
    import json
    from pathlib import Path
    metrics_path = Path("artifacts/model_evaluation/eval_report.json")
    model_name = "LightGBM v2.1"
    auc = 0.737
    threshold = 0.38
    status = "Live"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        auc = metrics.get('roc_auc', auc)
        threshold = metrics.get('threshold', threshold)
    bar_html = f"""
    <div class="sticky-bar">
        <span class="pill"><span class="live-dot"></span> {status}</span>
        <span class="pill">{model_name}</span>
        <span class="pill">AUC {auc:.3f}</span>
        <span class="pill">Threshold {threshold:.2f}</span>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)
