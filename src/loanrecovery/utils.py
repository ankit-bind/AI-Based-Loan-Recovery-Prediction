# ============================================================
# LOAN RECOVERY PROJECT — UTILITY FUNCTIONS
# ============================================================
"""
Reusable helper functions for the loan recovery pipeline.

Contains:
- Model save/load utilities
- Metric calculation
- Threshold optimization
- Plotting helpers
- YAML config loader
"""

import os
import sys
import json
import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)

from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException

logger = get_logger(__name__)


# ── YAML UTILITIES ───────────────────────────────────────────
def read_yaml(path: str) -> Dict:
    """
    Read YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary with YAML contents
    """
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"YAML loaded: {path}")
        return config
    except Exception as e:
        raise LoanRecoveryException(e, sys)


# ── MODEL UTILITIES ─────────────────────────────────────────
def save_object(
    path: str,
    obj: Any
) -> None:
    """
    Save Python object using joblib.

    Args:
        path: File path to save object
        obj: Python object to save
    """
    try:
        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )
        joblib.dump(obj, path)
        logger.info(f"Object saved: {path}")
    except Exception as e:
        raise LoanRecoveryException(e, sys)


def load_object(path: str) -> Any:
    """
    Load Python object using joblib.

    Args:
        path: File path of saved object

    Returns:
        Loaded Python object
    """
    try:
        obj = joblib.load(path)
        logger.info(f"Object loaded: {path}")
        return obj
    except Exception as e:
        raise LoanRecoveryException(e, sys)


def save_json(
    path: str,
    data: Dict
) -> None:
    """
    Save dictionary as JSON file.

    Args:
        path: File path to save JSON
        data: Dictionary to save
    """
    try:
        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON saved: {path}")
    except Exception as e:
        raise LoanRecoveryException(e, sys)


def load_json(path: str) -> Dict:
    """
    Load JSON file as dictionary.

    Args:
        path: File path of JSON file

    Returns:
        Dictionary with JSON contents
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"JSON loaded: {path}")
        return data
    except Exception as e:
        raise LoanRecoveryException(e, sys)


# ── METRICS UTILITIES ───────────────────────────────────────
def get_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict:
    """
    Calculate all classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities

    Returns:
        Dictionary with all metrics
    """
    try:
        metrics = {
            "accuracy" : round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc"  : round(roc_auc_score(y_true, y_prob), 4),
        }
        return metrics
    except Exception as e:
        raise LoanRecoveryException(e, sys)


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    min_recall: float = 0.55
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.

    Iterates through thresholds using precision-recall curve.
    Optimizes F1 score subject to a minimum recall constraint
    to prevent the degenerate solution of predicting everyone
    as positive (which maximizes raw recall at threshold=0).

    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        metric: Metric to optimize — always uses F1 with
                min_recall constraint regardless of this param
        min_recall: Minimum acceptable recall (default 0.55)

    Returns:
        Tuple of (best_threshold, best_score)
    """
    try:
        precisions, recalls, thresholds = (
            precision_recall_curve(y_true, y_prob)
        )

        # Align arrays: precision_recall_curve returns
        # len(thresholds) == len(precisions) - 1
        precisions = precisions[:-1]
        recalls = recalls[:-1]

        # Compute F1 for all thresholds
        f1_scores = (
            2 * precisions * recalls /
            (precisions + recalls + 1e-10)
        )

        # Only consider thresholds where recall >= min_recall
        valid_mask = recalls >= min_recall

        if valid_mask.any():
            # Among valid candidates, pick threshold with best F1
            valid_f1 = np.where(valid_mask, f1_scores, -1.0)
            best_idx = int(valid_f1.argmax())
        else:
            # Fallback: no threshold meets min_recall — pick
            # the one with highest recall overall
            logger.warning(
                f"No threshold achieves recall >= {min_recall:.2f}. "
                "Falling back to maximum recall threshold."
            )
            best_idx = int(recalls.argmax())

        best_threshold = float(thresholds[best_idx])
        best_score = float(f1_scores[best_idx])
        best_recall = float(recalls[best_idx])

        logger.info(
            f"Optimal threshold: {best_threshold:.3f} "
            f"(F1: {best_score:.3f} | Recall: {best_recall:.3f})"
        )

        return best_threshold, best_score

    except Exception as e:
        raise LoanRecoveryException(e, sys)


def classify_risk(
    prob: float,
    threshold_high: float = 0.70,
    threshold_med: float = 0.40
) -> str:
    """
    Classify borrower risk based on probability.

    Business rule layer:
    - HIGH   : prob >= 0.70 → Priority contact
    - MEDIUM : prob >= 0.40 → Standard follow-up
    - LOW    : prob < 0.40  → Consider write-off

    Args:
        prob: Recovery probability (0-1)
        threshold_high: High recovery threshold
        threshold_med: Medium recovery threshold

    Returns:
        Risk category string
    """
    if prob >= threshold_high:
        return "HIGH"
    elif prob >= threshold_med:
        return "MEDIUM"
    else:
        return "LOW"


# ── PLOTTING UTILITIES ───────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plot and optionally save confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot
        title: Plot title
    """
    try:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Not Recovered', 'Recovered'],
            yticklabels=['Not Recovered', 'Recovered']
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()

        if save_path:
            os.makedirs(
                os.path.dirname(save_path),
                exist_ok=True
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved: {save_path}")

        plt.show()
        plt.close()

    except Exception as e:
        raise LoanRecoveryException(e, sys)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        model_name: Name of the model
        save_path: Path to save plot
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr,
            color='#2196F3',
            linewidth=2,
            label=f'{model_name} (AUC = {auc:.3f})'
        )
        plt.plot(
            [0, 1], [0, 1],
            color='gray',
            linewidth=1,
            linestyle='--',
            label='Random Classifier'
        )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend()
        plt.tight_layout()

        if save_path:
            os.makedirs(
                os.path.dirname(save_path),
                exist_ok=True
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"ROC curve saved: {save_path}")

        plt.show()
        plt.close()

    except Exception as e:
        raise LoanRecoveryException(e, sys)


def evaluate_models(
    models: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    Train and evaluate multiple models.

    Args:
        models: Dictionary of model_name -> model_object
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold

    Returns:
        Dictionary with metrics for each model
    """
    try:
        results = {}

        for name, model in models.items():
            logger.info(f"Training: {name}")

            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)

            metrics = get_metrics(y_test, y_pred, y_prob)
            results[name] = metrics

            logger.info(
                f"{name} → "
                f"Recall: {metrics['recall']:.3f} | "
                f"AUC: {metrics['roc_auc']:.3f}"
            )

        return results

    except Exception as e:
        raise LoanRecoveryException(e, sys)


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce DataFrame memory by downcasting dtypes.

    Converts:
    - float64 → float32
    - int64   → int32

    Args:
        df: Input DataFrame

    Returns:
        Memory-optimized DataFrame
    """
    try:
        before = df.memory_usage(deep=True).sum() / 1024**2

        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')

        after = df.memory_usage(deep=True).sum() / 1024**2
        saved = (before - after) / before * 100

        logger.info(
            f"Memory: {before:.1f}MB → {after:.1f}MB "
            f"({saved:.0f}% saved)"
        )

        return df

    except Exception as e:
        raise LoanRecoveryException(e, sys)
