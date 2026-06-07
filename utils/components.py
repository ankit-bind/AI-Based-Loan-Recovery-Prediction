import streamlit as st
from pathlib import Path
import json

# ── Risk Tier Utilities ──
def get_risk_tier(probability: float):
    """Map probability to risk tier and metadata."""
    if probability < 0.25:
        return {
            "tier": "Critical",
            "badge_class": "badge-critical",
            "color": "#E24B4A",
            "bg": "rgba(226, 75, 74, 0.12)",
            "action": "Escalate to legal team · Consider structured settlement"
        }
    elif probability < 0.50:
        return {
            "tier": "Low",
            "badge_class": "badge-low",
            "color": "#D4A24C",
            "bg": "rgba(186, 117, 23, 0.12)",
            "action": "Active management required · Schedule collection call"
        }
    elif probability < 0.75:
        return {
            "tier": "Moderate",
            "badge_class": "badge-moderate",
            "color": "#4A9FE6",
            "bg": "rgba(24, 95, 165, 0.12)",
            "action": "Standard monitoring · Routine payment follow-up"
        }
    else:
        return {
            "tier": "High",
            "badge_class": "badge-high",
            "color": "#6BBF2A",
            "bg": "rgba(59, 109, 17, 0.12)",
            "action": "Self-service track · Continue regular servicing"
        }


def risk_badge(probability: float):
    """Return HTML string for a risk badge."""
    tier = get_risk_tier(probability)
    return f"""
    <span class="risk-badge {tier['badge_class']}">
        {tier['tier']}
    </span>
    """


def render_metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Render a custom metric card."""
    delta_html = ""
    if delta:
        color = "#6BBF2A" if delta_color == "normal" else "#E24B4A" if delta_color == "inverse" else "#4A9FE6"
        delta_html = f'<div class="delta" style="color:{color}">{delta}</div>'
    html = f"""
    <div class="metric-card">
        <div class="val">{value}</div>
        <div class="lbl">{label}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_borrower_card(loan_id: str, loan_type: str, outstanding: str, probability: float,
                         shap_factors: list, confidence: str = "High"):
    """
    Render a borrower intelligence card.
    shap_factors: list of dicts like {"name": "Debt-to-income", "value": "0.89", "direction": "up", "pct": 85}
    """
    tier = get_risk_tier(probability)
    initials = "LR"  # generic
    shap_html = ""
    for f in shap_factors[:3]:
        direction_color = "#E24B4A" if f["direction"] == "up" else "#6BBF2A"
        arrow = "↑" if f["direction"] == "up" else "↓"
        bar_color = "#E24B4A" if f["direction"] == "up" else "#3B6D11"
        shap_html += f"""<div class="shap-row"><span style="color:{direction_color}; font-weight:600;">{arrow}</span><span style="flex:1;">{f['name']}: {f['value']}</span><div class="bar-bg"><div class="bar-fill" style="width:{f['pct']}%; background:{bar_color};"></div></div></div>"""
    card = f"""<div class="borrower-card"><div class="header"><div class="avatar" style="background:{tier['bg']}; color:{tier['color']};">{initials}</div><div class="meta"><p class="title">Loan #{loan_id}</p><p class="subtitle">{loan_type} · {outstanding} outstanding</p></div><div class="status"><span class="risk-badge {tier['badge_class']}">{tier['tier']}</span><p class="prob">Predicted recovery: {probability*100:.0f}%</p></div></div><div class="bp-divider"></div><div style="display:flex; gap:10px; margin:8px 0;"><div class="metric-card" style="flex:1;"><div class="val" style="font-size:16px; color:{tier['color']};">{probability*100:.0f}%</div><div class="lbl">Recovery prob.</div></div><div class="metric-card" style="flex:1;"><div class="val" style="font-size:16px;">{tier['tier']}</div><div class="lbl">Risk tier</div></div><div class="metric-card" style="flex:1;"><div class="val" style="font-size:16px;">{confidence}</div><div class="lbl">Confidence</div></div></div><div class="bp-divider"></div><p style="font-size:12px; font-weight:500; color:#F0F2F6; margin-bottom:6px;">Top 3 risk factors</p>{shap_html}<div class="bp-divider"></div><div class="action-box"><span class="label">Recommended action: </span><span class="action">{tier['action']}</span></div></div>"""
    st.markdown(card, unsafe_allow_html=True)


def render_health_banner(total_loans: int, critical_count: int, last_updated: str):
    """Render the portfolio health banner."""
    html = f"""
    <div class="health-banner">
        <span style="font-size:16px;"></span>
        <span><strong>{total_loans:,}</strong> loans monitored · <strong style="color:#E24B4A;">{critical_count}</strong> critical · Last updated {last_updated}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_insight_card(text: str):
    """Render AI insight card."""
    html = f"""
    <div class="insight-card">
        <div class="insight-title">AI Insight</div>
        <div>{text}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_pipeline_stage(icon: str, title: str, description: str, border_color: str = "#2A3B52"):
    """Render a pipeline stage card."""
    html = f"""
    <div class="pipeline-stage" style="border-color: {border_color};">
        <div class="stage-title">{icon} {title}</div>
        <p class="stage-desc">{description}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def empty_state_upload():
    """Render empty state for upload zone."""
    html = """
    <div class="empty-state">
        <div style="font-size:32px; margin-bottom:8px;"></div>
        <p style="font-weight:500; color:#F0F2F6; margin:8px 0 4px; font-size:14px;">Upload a loan portfolio CSV</p>
        <p style="font-size:12px; color:#B8C4D4; margin-bottom:14px;">Accepts .csv files up to 50MB · Up to 10,000 rows per batch</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_verdict_card(verdict: str, explanation: str, severity: str = "info"):
    """
    Render a plain-language model verdict card.
    severity: info, success, warning, error
    """
    border_map = {
        "info": "rgba(24, 95, 165, 0.3)",
        "success": "rgba(59, 109, 17, 0.3)",
        "warning": "rgba(186, 117, 23, 0.3)",
        "error": "rgba(226, 75, 74, 0.3)"
    }
    bg_map = {
        "info": "rgba(24, 95, 165, 0.08)",
        "success": "rgba(59, 109, 17, 0.08)",
        "warning": "rgba(186, 117, 23, 0.08)",
        "error": "rgba(226, 75, 74, 0.08)"
    }
    html = f"""
    <div style="background: {bg_map[severity]}; border: 0.5px solid {border_map[severity]}; border-radius: 12px; padding: 14px 16px; margin: 10px 0;">
        <p style="font-size:13px; font-weight:500; color:#F0F2F6; margin-bottom:6px;">{verdict}</p>
        <p style="font-size:12px; color:#F0F2F6; margin:0;">{explanation}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_shap_factors(factors: list):
    """
    Render inline SHAP factors (positive vs negative).
    factors: list of dicts {"name": str, "value": str, "impact": float, "direction": "up"|"down"}
    """
    html = '<div style="margin: 10px 0;">'
    for f in factors:
        color = "#E24B4A" if f["direction"] == "up" else "#6BBF2A"
        arrow = "↑" if f["direction"] == "up" else "↓"
        bar_color = "#E24B4A" if f["direction"] == "up" else "#3B6D11"
        pct = min(abs(f["impact"]) * 100, 100)
        html += f"""<div class="shap-row"><span style="color:{color}; font-weight:600;">{arrow}</span><span style="flex:1;">{f['name']}: {f['value']}</span><div class="bar-bg"><div class="bar-fill" style="width:{pct}%; background:{bar_color};"></div></div></div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_section_label(text: str):
    st.markdown(f"""
    <p style="font-size:10px; font-weight:500; color:#B8C4D4; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:4px;">
        {text}
    </p>
    """, unsafe_allow_html=True)


def render_section_title(text: str):
    st.markdown(f"""
    <p style="font-size:18px; font-weight:500; color:#FFFFFF; margin-bottom:1rem;">
        {text}
    </p>
    """, unsafe_allow_html=True)
