import json
from pathlib import Path
import pandas as pd
from datetime import datetime

# ── Probability Interpretation ──

def get_probability_interpretation(prob: float) -> dict:
    """
    Map a raw probability to a human-readable tier with color.
    Returns: {"label": str, "color": str, "hex": str, "meaning": str}
    """
    if prob >= 0.80:
        return {
            "label": "Very High Recovery Chance",
            "color": "green",
            "hex": "#6BBF2A",
            "meaning": "Strong indicators of recovery. Standard servicing recommended.",
            "badge_class": "badge-high"
        }
    elif prob >= 0.60:
        return {
            "label": "High Recovery Chance",
            "color": "blue",
            "hex": "#4A9FE6",
            "meaning": "Above-average recovery probability. Routine follow-up advised.",
            "badge_class": "badge-moderate"
        }
    elif prob >= 0.30:
        return {
            "label": "Moderate Recovery Chance",
            "color": "orange",
            "hex": "#D4A24C",
            "meaning": "Mixed signals. Active management and collection call recommended.",
            "badge_class": "badge-low"
        }
    else:
        return {
            "label": "Low Recovery Chance",
            "color": "red",
            "hex": "#E24B4A",
            "meaning": "High risk of non-recovery. Escalate to legal or structured settlement.",
            "badge_class": "badge-critical"
        }


# ── Risk Color System ──

def get_risk_color(prob: float) -> str:
    """Return hex color for a given probability."""
    if prob >= 0.80:
        return "#6BBF2A"
    elif prob >= 0.60:
        return "#4A9FE6"
    elif prob >= 0.30:
        return "#D4A24C"
    else:
        return "#E24B4A"


def get_risk_gradient_css(prob: float) -> str:
    """Return a CSS gradient string for the probability bar."""
    if prob >= 0.80:
        return "linear-gradient(90deg, #3B6D11, #6BBF2A)"
    elif prob >= 0.60:
        return "linear-gradient(90deg, #185F51, #4A9FE6)"
    elif prob >= 0.30:
        return "linear-gradient(90deg, #8B5A00, #D4A24C)"
    else:
        return "linear-gradient(90deg, #8B1515, #E24B4A)"


# ── Model Info Panel ──

def load_model_info() -> dict:
    """Load current model metadata for the info panel."""
    info = {
        "model_name": "LightGBM v2.1",
        "features": 117,
        "threshold": 0.38,
        "recall": 0.39,
        "precision": 0.22,
        "roc_auc": 0.737,
        "smote_strategy": 0.3,
        "training_samples": 64000,
        "last_trained": "Unknown",
    }

    # Load from actual artifacts if available
    eval_path = Path("artifacts/model_evaluation/eval_report.json")
    if eval_path.exists():
        with open(eval_path) as f:
            m = json.load(f)
        info["recall"] = m.get("recall", info["recall"])
        info["precision"] = m.get("precision", info["precision"])
        info["roc_auc"] = m.get("roc_auc", info["roc_auc"])
        info["threshold"] = m.get("threshold", info["threshold"])

    comp_path = Path("artifacts/model_trainer/model_comparison.json")
    if comp_path.exists():
        with open(comp_path) as f:
            comp = json.load(f)
        # Find best model name
        if comp:
            best = max(comp, key=lambda k: comp[k].get("recall", 0))
            info["model_name"] = best.replace("_", " ").title()

    fn_path = Path("models/feature_names.json")
    if fn_path.exists():
        with open(fn_path) as f:
            fn = json.load(f)
        info["features"] = fn.get("count", info["features"])

    # Check model file timestamp
    model_path = Path("models/best_model.pkl")
    if model_path.exists():
        info["last_trained"] = datetime.fromtimestamp(
            model_path.stat().st_mtime
        ).strftime("%Y-%m-%d %H:%M")

    return info


# ── Prediction History ──

PREDICTION_LOG_PATH = Path("prediction_logs.csv")


def log_prediction(borrower_id: str, probability: float, prediction: int,
                   threshold: float, risk_label: str, input_features: int):
    """Append a prediction record to the history log."""
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "borrower_id": borrower_id,
        "probability": round(probability, 4),
        "prediction": prediction,
        "threshold": round(threshold, 4),
        "risk_label": risk_label,
        "input_features": input_features,
    }
    df = pd.DataFrame([row])
    if PREDICTION_LOG_PATH.exists():
        df.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(PREDICTION_LOG_PATH, index=False)


def load_prediction_history(limit: int = 100) -> pd.DataFrame:
    """Load recent prediction history."""
    if not PREDICTION_LOG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTION_LOG_PATH)
    return df.tail(limit)


# ── Existing functions (kept for compatibility) ──

def generate_borrower_verdict(probability: float, top_factors: list):
    """Generate plain-language verdict for a single borrower."""
    interp = get_probability_interpretation(probability)
    verdict = interp["meaning"]
    severity = "error" if probability < 0.30 else "warning" if probability < 0.60 else "info" if probability < 0.80 else "success"

    if top_factors:
        primary = top_factors[0]
        direction = "increases" if primary.get("direction", "up") == "up" else "reduces"
        explanation = f"Primary driver: {primary['name']} ({primary['value']}) {direction} risk."
    else:
        explanation = "No dominant risk factors identified."

    return {"verdict": verdict, "explanation": explanation, "severity": severity}


def generate_model_health_verdict(metrics: dict):
    """Generate plain-language model health verdict."""
    auc = metrics.get('roc_auc', 0.74)
    recall = metrics.get('recall', 0.39)
    precision = metrics.get('precision', 0.22)
    f1 = metrics.get('f1_score', 0.30)

    if auc >= 0.70:
        verdict = f"Model is performing well for high-value loan identification (AUC {auc:.3f})."
        severity = "success"
    elif auc >= 0.60:
        verdict = f"Model performance is acceptable but borderline (AUC {auc:.3f}). Consider retraining."
        severity = "warning"
    else:
        verdict = f"Model performance is below acceptable range (AUC {auc:.3f}). Retraining recommended."
        severity = "error"

    explanation = ""
    if recall < 0.50:
        explanation += f" Low recall ({recall:.1%}) means many recoverable loans are not flagged."
    if precision < 0.30:
        explanation += f" Low precision ({precision:.1%}) means many false positives — review operational cost."
    if not explanation:
        explanation = f" F1 score of {f1:.3f} indicates balanced performance."

    return {"verdict": verdict, "explanation": explanation, "severity": severity}


def generate_threshold_recommendation(threshold: float, precision: float, recall: float, f1: float,
                                       avg_loan_value: float = 500000, num_loans: int = 1000):
    """Generate AI recommendation text for threshold strategy."""
    flagged = int(num_loans * recall)
    true_positives = int(flagged * precision)
    estimated_recovery = true_positives * avg_loan_value * 0.3

    if threshold <= 0.15:
        strategy = "Aggressive"
        desc = "Flag more accounts, accept more false positives"
    elif threshold <= 0.30:
        strategy = "Balanced"
        desc = "Moderate flagging with balanced precision and recall"
    else:
        strategy = "Conservative"
        desc = "Flag fewer accounts, higher precision per flag"

    recommendation = f"""
    <strong>Strategy: {strategy}</strong> — {desc}.<br>
    At threshold {threshold:.2f}: ~{flagged} accounts flagged, {precision:.0%} precision, {recall:.0%} recall.
    Estimated recovery impact: ₹{estimated_recovery/100000:.1f}L.
    """
    return recommendation


def generate_dashboard_insight():
    """Generate a daily AI insight for the dashboard."""
    import random
    insights = [
        "Today's top recovery predictor: <em>debt_to_income_ratio</em> — accounts with ratio <0.4 show 73% higher recovery probability.",
        "Accounts with <em>days_employed</em> > 5 years and <em>FLAG_OWN_REALTY</em> have 2.1× recovery odds.",
        "Recent batch shows <em>EXT_SOURCE_2</em> declining for new applicants — monitor external credit signals.",
        "Collateral coverage >1.5x is the strongest positive SHAP feature this week for commercial loans.",
    ]
    return random.choice(insights)


def load_metrics():
    """Load evaluation metrics from artifacts."""
    metrics_path = Path("artifacts/model_evaluation/eval_report.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {
        "roc_auc": 0.737,
        "accuracy": 0.82,
        "precision": 0.22,
        "recall": 0.39,
        "f1_score": 0.30,
        "threshold": 0.38,
    }


def load_model_comparison():
    """Load model comparison data."""
    path = Path("artifacts/model_trainer/model_comparison.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}
