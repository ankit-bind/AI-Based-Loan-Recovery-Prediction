import os
import sys
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException
from src.loanrecovery.utils import save_json

logger = get_logger(__name__)

class ModelEvaluation:
    def __init__(self, config):
        self.config = config

    def initiate_model_evaluation(self, engineered_data_path: str):
        logger.info('Loan Recovery Model Evaluation Started')
        try:
            # Load saved test set (same as model trainer used)
            test_data_path = os.path.join(self.config.root_dir, "..", "model_trainer", "test_data.pkl")
            test_data_path = os.path.abspath(test_data_path)
            
            if not os.path.exists(test_data_path):
                # Fallback: use the engineered data and do same split
                logger.warning("Saved test set not found, using engineered data split")
                df = pd.read_csv(engineered_data_path)
                X = df.drop(columns=[self.config.target_column])
                y = df[self.config.target_column]
                test_size = int(len(df) * self.config.test_size)
                X_test = X.tail(test_size)
                y_test = y.tail(test_size)
            else:
                test_data = joblib.load(test_data_path)
                X_test = test_data["X_test"]
                y_test = test_data["y_test"]
            
            # Feature alignment validation
            feature_path = os.path.join('models', 'feature_names.json')
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    feature_info = json.load(f)
                    expected_features = feature_info['features']
                    # Ensure columns exist
                    missing_features = [col for col in expected_features if col not in X_test.columns]
                    if missing_features:
                        logger.warning(f'Missing features in test set: {missing_features}')
                    else:
                        X_test = X_test[expected_features]
                        logger.info('Feature alignment completed successfully')
            
            # Remaining null validation
            remaining_nulls = X_test.isnull().sum().sum()
            logger.info(f'Remaining null values: {remaining_nulls}')
            
            # Load model and threshold
            model_path = os.path.join('models', 'best_model.pkl')
            threshold_path = os.path.join('models', 'threshold.pkl')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model not found at {model_path}')
            if not os.path.exists(threshold_path):
                raise FileNotFoundError(f'Threshold file not found at {threshold_path}')
            
            model = joblib.load(model_path)
            threshold = joblib.load(threshold_path)
            
            # Apply scaler ONLY if the best model is a linear model (e.g. LogisticRegression)
            # Tree-based models (LightGBM, XGBoost, RandomForest) must NOT be scaled
            scaler_path = os.path.join('models', "scaler.pkl")
            is_linear_model = hasattr(model, 'coef_')  # True for LogisticRegression, LinearSVC, etc.
            if os.path.exists(scaler_path) and is_linear_model:
                scaler = joblib.load(scaler_path)
                X_test_transformed = scaler.transform(X_test)
                logger.info('Applied scaler (linear model detected)')
            else:
                X_test_transformed = X_test
                if os.path.exists(scaler_path) and not is_linear_model:
                    logger.info('Skipped scaler (tree-based model — scaling not needed)')
            
            # Predict
            y_proba = model.predict_proba(X_test_transformed)[:, 1]
            y_pred = (y_proba >= threshold).astype(int) if isinstance(threshold, float) else model.predict(X_test_transformed)
            
            # Probability range logging
            logger.info(f'Probability range: {y_proba.min():.4f} to {y_proba.max():.4f}')
            
            # Evaluate
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }
            
            # Evaluation summary logging
            logger.info(
                f'Accuracy: {metrics["accuracy"]:.4f} | '
                f'Precision: {metrics["precision"]:.4f} | '
                f'Recall: {metrics["recall"]:.4f} | '
                f'F1: {metrics["f1_score"]:.4f} | '
                f'ROC-AUC: {metrics["roc_auc"]:.4f}'
            )
            
            # Save metrics
            report_path = os.path.join(self.config.root_dir, "eval_report.json")
            save_json(report_path, metrics)
            logger.info(f'Metrics saved to {report_path}')
            
            # Classification report
            report = classification_report(
                y_test, y_pred,
                target_names=['Not Recovered', 'Recovered'],
                output_dict=True
            )
            classification_report_path = os.path.join(self.config.root_dir, 'classification_report.json')
            save_json(classification_report_path, report)
            logger.info('Classification report saved successfully')
            
            # ROC Curve Plot
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label='ROC Curve')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            roc_path = os.path.join(self.config.root_dir, 'roc_curve.png')
            plt.savefig(roc_path, bbox_inches='tight', dpi=150)
            plt.close()
            logger.info(f'ROC curve saved to {roc_path}')
            
            # Confusion Matrix Heatmap
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Not Recovered', 'Recovered'],
                        yticklabels=['Not Recovered', 'Recovered'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            cm_path = os.path.join(self.config.root_dir, 'confusion_matrix.png')
            plt.savefig(cm_path, bbox_inches='tight', dpi=150)
            plt.close()
            logger.info(f'Confusion matrix saved to {cm_path}')
            
            # Precision-Recall / Threshold Curve
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precisions[:-1], label='Precision')
            plt.plot(thresholds, recalls[:-1], label='Recall')
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Threshold vs Precision/Recall')
            plt.legend()
            pr_path = os.path.join(self.config.root_dir, 'pr_curve.png')
            plt.savefig(pr_path, bbox_inches='tight', dpi=150)
            plt.close()
            logger.info(f'PR curve saved to {pr_path}')
            
            return metrics

        except Exception as e:
            raise LoanRecoveryException(e, sys)
