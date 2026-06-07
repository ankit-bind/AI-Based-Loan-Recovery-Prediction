import plotly.graph_objects as go
import plotly.express as px

# ── Fintech Plotly Theme ──
FINTECH_COLORS = ['#378ADD', '#3B6D11', '#D4A24C', '#E24B4A', '#534AB7']
RISK_COLORS = {
    "Critical": '#E24B4A',
    "Low": '#D4A24C',
    "Moderate": '#4A9FE6',
    "High": '#6BBF2A'
}

fintech_layout = dict(
    font=dict(family="Inter, sans-serif", size=12, color="#F0F2F6"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.06)",
        linecolor="#2A3B52", tickfont=dict(size=11, color="#B8C4D4")
    ),
    yaxis=dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.06)",
        linecolor="#2A3B52", tickfont=dict(size=11, color="#B8C4D4")
    ),
    colorway=FINTECH_COLORS,
)

plotly_config = dict(
    displayModeBar=True,
    displaylogo=False,
    modeBarButtonsToRemove=["lasso2d", "select2d", "autoScale2d"]
)


def apply_fintech_theme(fig: go.Figure):
    # Use fig.layout.update() to bypass update_layout()'s named 'legend'
    # parameter which causes collision when unpacking **fintech_layout
    fig.layout.update(fintech_layout)
    return fig


FINTECH_LEGEND = dict(
    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    font=dict(size=11, color="#F0F2F6")
)


def risk_donut_chart(counts: dict):
    """
    counts: {"Critical": int, "Low": int, "Moderate": int, "High": int}
    """
    labels = list(counts.keys())
    values = list(counts.values())
    colors = [RISK_COLORS.get(l, "#378ADD") for l in labels]
    total = sum(values)
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.55,
        marker_colors=colors,
        textinfo="label+percent",
        textfont_size=11,
        insidetextorientation="horizontal"
    )])
    fig = apply_fintech_theme(fig)
    fig.update_layout(
        title=dict(text="Risk Tier Distribution", x=0, xanchor="left"),
        annotations=[dict(text=f"{total}", x=0.5, y=0.5, font_size=18, font_color="#F0F2F6", showarrow=False)],
        showlegend=False,
    )
    return fig


def recovery_histogram(values: list, threshold: float = 0.38):
    """Recovery probability histogram with threshold line and color-coded bins."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values, nbinsx=40,
        marker_color="#378ADD",
        opacity=0.85,
        name="Borrowers"
    ))
    fig.add_vline(
        x=threshold, line_width=2, line_dash="dash", line_color="#F0F2F6",
        annotation_text="Threshold", annotation_position="top",
        annotation_font_size=11, annotation_font_color="#F0F2F6"
    )
    fig = apply_fintech_theme(fig)
    fig.update_layout(
        title=dict(text="Recovery Probability Distribution", x=0, xanchor="left"),
        xaxis_title="Probability", yaxis_title="Count",
        bargap=0.05,
    )
    return fig


def roc_curve_chart(fpr, tpr, auc: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", fill="tozeroy",
        line=dict(color="#378ADD", width=2),
        fillcolor="rgba(55, 138, 221, 0.15)",
        name=f"Model (AUC = {auc:.3f})"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dash"),
        name="Random classifier"
    ))
    fig.add_annotation(
        x=0.65, y=0.25, text=f"AUC = {auc:.3f}",
        showarrow=False, font=dict(size=13, color="#F0F2F6"),
        bgcolor="rgba(20, 29, 43, 0.8)", bordercolor="#2A3B52", borderwidth=1, borderpad=4,
    )
    fig = apply_fintech_theme(fig)
    fig.update_layout(
        title=dict(text="ROC Curve", x=0, xanchor="left"),
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
    )
    return fig


def precision_recall_curve(precisions, recalls, thresholds, f1s, highlight_threshold: float = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recalls, y=precisions, mode="lines",
        line=dict(color="#378ADD", width=2), name="Precision-Recall"
    ))
    if highlight_threshold is not None:
        # Find nearest
        idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - highlight_threshold))
        fig.add_trace(go.Scatter(
            x=[recalls[idx]], y=[precisions[idx]], mode="markers",
            marker=dict(color="#E24B4A", size=12, symbol="circle"),
            name=f"Current threshold ({highlight_threshold:.2f})"
        ))
    fig = apply_fintech_theme(fig)
    fig.update_layout(
        title=dict(text="Precision-Recall Curve", x=0, xanchor="left"),
        xaxis_title="Recall", yaxis_title="Precision",
    )
    return fig


def confusion_matrix_chart(cm_data: dict):
    """
    cm_data: dict with keys TN, FP, FN, TP
    """
    z = [[cm_data["TN"], cm_data["FP"]], [cm_data["FN"], cm_data["TP"]]]
    total = cm_data["TN"] + cm_data["FP"] + cm_data["FN"] + cm_data["TP"]
    text = [
        [f"{cm_data['TN']}\n({cm_data['TN']/total*100:.1f}%)", f"{cm_data['FP']}\n({cm_data['FP']/total*100:.1f}%)"],
        [f"{cm_data['FN']}\n({cm_data['FN']/total*100:.1f}%)", f"{cm_data['TP']}\n({cm_data['TP']/total*100:.1f}%)"]
    ]
    # Dark colorscale: low values = dark, high values = brighter blue
    colorscale = [[0, "#0A1628"], [0.25, "#0F2137"], [0.5, "#163A6B"], [0.75, "#2E6FAD"], [1, "#5CB0FF"]]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=["Predicted Not Recovered", "Predicted Recovered"],
        y=["Actual Not Recovered", "Actual Recovered"],
        colorscale=colorscale,
        zmin=0, zmax=total, zauto=False,
        text=text, texttemplate="%{text}", textfont=dict(size=12, color="#FFFFFF"),
        hoverongaps=False,
        hoverlabel=dict(bgcolor="#141D2B", font=dict(color="#F0F2F6", size=12), bordercolor="#2A3B52"),
    ))
    # Highlight FN cell
    fig.add_shape(
        type="rect", x0=-0.4, y0=0.6, x1=0.4, y1=1.4,
        line=dict(color="#E24B4A", width=2),
        fillcolor="rgba(226, 75, 74, 0.15)", layer="above"
    )
    fig.add_annotation(
        x=0, y=1, text="Missed recovery", showarrow=False,
        font=dict(size=10, color="#FF6B6B"), yshift=-22
    )
    fig = apply_fintech_theme(fig)
    # Force colorscale again after theme apply
    fig.update_traces(
        selector=dict(type='heatmap'),
        colorscale=colorscale,
        zmin=0, zmax=total, zauto=False
    )
    fig.update_layout(
        title=dict(text="Confusion Matrix", x=0, xanchor="left"),
    )
    return fig


def feature_importance_chart(df, x_col="importance", y_col="feature", top_n=20):
    """Horizontal bar chart for feature importance."""
    df = df.head(top_n).sort_values(by=x_col, ascending=True)
    fig = go.Figure(go.Bar(
        x=df[x_col], y=df[y_col], orientation="h",
        marker=dict(color="#378ADD", line=dict(width=0)),
        text=df[x_col].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
        textfont=dict(size=10, color="#F0F2F6"),
    ))
    fig.add_vline(x=df[x_col].mean(), line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.2)")
    fig.add_annotation(x=df[x_col].mean(), y=df[y_col].iloc[-1], text="Average", showarrow=False,
                       font=dict(size=10, color="#B8C4D4"), yshift=14)
    fig = apply_fintech_theme(fig)
    fig.update_layout(
        title=dict(text=f"Top {top_n} Important Features", x=0, xanchor="left"),
        xaxis_title="Importance", yaxis_title="",
        margin=dict(l=140, r=40, t=50, b=40),
    )
    return fig


def gauge_chart(probability: float, threshold: float = 0.38):
    """Semicircle gauge for recovery probability."""
    tier = "Critical" if probability < 0.25 else "Low" if probability < 0.50 else "Moderate" if probability < 0.75 else "High"
    colors = ["#E24B4A", "#D4A24C", "#4A9FE6", "#6BBF2A"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number=dict(suffix="%", font=dict(size=28, color="#F0F2F6")),
        title=dict(text="Recovery Probability", font=dict(size=13, color="#B8C4D4")),
        delta=dict(reference=threshold * 100, relative=False, valueformat=".1f",
                   suffix="%", font=dict(size=12)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#2A3B52"),
            bar=dict(color=colors[["Critical", "Low", "Moderate", "High"].index(tier)], thickness=0.75),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, 25], color="rgba(226, 75, 74, 0.15)"),
                dict(range=[25, 50], color="rgba(186, 117, 23, 0.15)"),
                dict(range=[50, 75], color="rgba(24, 95, 165, 0.15)"),
                dict(range=[75, 100], color="rgba(59, 109, 17, 0.15)"),
            ],
            threshold=dict(
                line=dict(color="#F0F2F6", width=2),
                thickness=0.85, value=threshold * 100
            ),
        )
    ))
    fig = apply_fintech_theme(fig)
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=280,
    )
    return fig


def threshold_sensitivity_chart(thresholds, precisions, recalls, f1s, current_threshold: float):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=recalls, mode="lines", name="Recall", line=dict(color="#4A9FE6")))
    fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode="lines", name="Precision", line=dict(color="#378ADD")))
    fig.add_trace(go.Scatter(x=thresholds, y=f1s, mode="lines", name="F1-Score", line=dict(color="#6BBF2A")))
    fig.add_vline(x=current_threshold, line_width=2, line_dash="dash", line_color="#E24B4A",
                  annotation_text="Current", annotation_position="top",
                  annotation_font_size=11, annotation_font_color="#E24B4A")
    fig = apply_fintech_theme(fig)
    fig.update_layout(
        title=dict(text="Threshold Sensitivity", x=0, xanchor="left"),
        xaxis_title="Threshold", yaxis_title="Score",
    )
    return fig


def model_comparison_chart(comparison_df):
    """Side-by-side model comparison bar chart."""
    fig = go.Figure()
    metrics = ["roc_auc", "f1", "recall", "precision"]
    metric_labels = ["ROC-AUC", "F1", "Recall", "Precision"]
    for i, m in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric_labels[i],
            x=comparison_df["Model"],
            y=comparison_df[m],
            marker_color=FINTECH_COLORS[i],
        ))
    fig = apply_fintech_theme(fig)
    # Use fig.layout.update() instead of fig.update_layout() to avoid
    # Plotly's internal collision on the 'legend' keyword argument
    fig.layout.update(dict(
        barmode="group",
        title=dict(text="Model Comparison", x=0, xanchor="left"),
        yaxis=dict(title="Score"),
        legend=FINTECH_LEGEND,
    ))
    return fig


def sparkline(data: list, color: str = "#E24B4A"):
    """Tiny sparkline trace."""
    fig = go.Figure(go.Scatter(
        x=list(range(len(data))), y=data,
        mode="lines", fill="tozeroy",
        line=dict(color=color, width=1.5),
        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.15,)}".replace("'", ""),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=40, width=120,
        showlegend=False,
    )
    return fig
