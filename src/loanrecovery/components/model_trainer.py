import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException
from src.loanrecovery.utils import save_object, save_json, get_metrics, optimize_threshold
from src.loanrecovery.pipeline.unified_preprocessor import UnifiedPreprocessor

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def split_data(self, df):
        """Split data into train and test sets."""
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance."""
        logger.info("Applying SMOTE (sampling_strategy=0.3)")
        smote = SMOTE(sampling_strategy=0.3, random_state=self.config.random_state)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        logger.info(
            f"After SMOTE - Class 0: {(y_train_smote==0).sum()}, "
            f"Class 1: {(y_train_smote==1).sum()}"
        )
        return X_train_smote, y_train_smote

    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression with scaling and F1-optimal threshold."""
        logger.info("Training Logistic Regression")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=self.config.random_state
        )
        model.fit(X_train_scaled, y_train)
        
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        threshold, _ = optimize_threshold(y_test, y_proba, metric="f1")
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = get_metrics(y_test, y_pred, y_proba)
        metrics['threshold'] = threshold
        
        return model, scaler, metrics

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest with F1-optimal threshold."""
        logger.info("Training Random Forest")
        
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=self.config.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold, _ = optimize_threshold(y_test, y_proba, metric="f1")
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = get_metrics(y_test, y_pred, y_proba)
        metrics['threshold'] = threshold
        
        return model, None, metrics

    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM with F1-optimal threshold."""
        logger.info("Training LightGBM")
        
        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold, _ = optimize_threshold(y_test, y_proba, metric="f1")
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = get_metrics(y_test, y_pred, y_proba)
        metrics['threshold'] = threshold
        
        return model, None, metrics

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost with F1-optimal threshold."""
        logger.info("Training XGBoost")
        
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            scale_pos_weight=11,
            eval_metric='logloss',
            random_state=self.config.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_test)[:, 1]
        threshold, _ = optimize_threshold(y_test, y_proba, metric="f1")
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = get_metrics(y_test, y_pred, y_proba)
        metrics['threshold'] = threshold
        
        return model, None, metrics

    # ── Plotting utilities ──────────────────────────────────────────────

    def save_roc_curve(self, y_test, y_proba, path):
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f'ROC curve saved to {path}')

    def save_confusion_matrix_plot(self, y_test, y_pred, path):
        import seaborn as sns
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Recovered', 'Recovered'],
                    yticklabels=['Not Recovered', 'Recovered'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f'Confusion matrix saved to {path}')

    def save_pr_curve(self, y_test, y_proba, path):
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, precisions[:-1], label='Precision')
        plt.plot(thresholds, recalls[:-1], label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold vs Precision/Recall')
        plt.legend()
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        logger.info(f'PR curve saved to {path}')

    def save_feature_importance(self, model, feature_names, path):
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values(by='importance', ascending=False)
            importance_df.to_csv(path, index=False)
            logger.info(f'Feature importance saved to {path}')

    # ── Main orchestrator ─────────────────────────────────────────────────

    def initiate_model_trainer(self, engineered_data_path: str):
        logger.info('Loan Recovery Model Training Started')
        try:
            # ═══════════════════════════════════════════════════════════════
            #  PHASE 1: Load raw engineered data & fit UnifiedPreprocessor
            # ═══════════════════════════════════════════════════════════════
            raw_df = pd.read_csv(engineered_data_path)
            logger.info(f"Loaded engineered dataset: {raw_df.shape}")
            
            preprocessor = UnifiedPreprocessor(
                target_column=self.config.target_column,
                apply_scaler=False  # scaler handled per-model below
            )
            # fit_transform learns all parameters and produces the clean matrix
            df_processed = preprocessor.fit_transform(raw_df)
            logger.info(f"Preprocessor fitted. Processed shape: {df_processed.shape}")
            
            # Save the preprocessor artifact — CRITICAL for inference consistency
            preprocessor_path = os.path.join('models', 'preprocessor.pkl')
            preprocessor.save(preprocessor_path)
            
            # ═══════════════════════════════════════════════════════════════
            #  PHASE 2: Train / test split on the PREPROCESSED data
            # ═══════════════════════════════════════════════════════════════
            X_train, X_test, y_train, y_test = self.split_data(df_processed)
            
            test_data_path = os.path.join(self.config.root_dir, "test_data.pkl")
            joblib.dump({"X_test": X_test, "y_test": y_test}, test_data_path)
            logger.info(f"Saved test set to {test_data_path}")
            
            # Apply SMOTE
            X_train_smote, y_train_smote = self.apply_smote(X_train, y_train)
            
            # ═══════════════════════════════════════════════════════════════
            #  PHASE 3: Train all candidate models
            # ═══════════════════════════════════════════════════════════════
            results = {}
            
            lr_model, lr_scaler, lr_metrics = self.train_logistic_regression(
                X_train_smote, y_train_smote, X_test, y_test
            )
            results['logistic_regression'] = lr_metrics
            logger.info(f"LR Metrics: {lr_metrics}")
            
            rf_model, _, rf_metrics = self.train_random_forest(
                X_train_smote, y_train_smote, X_test, y_test
            )
            results['random_forest'] = rf_metrics
            logger.info(f"RF Metrics: {rf_metrics}")
            
            lgbm_model, _, lgbm_metrics = self.train_lightgbm(
                X_train_smote, y_train_smote, X_test, y_test
            )
            results['lightgbm'] = lgbm_metrics
            logger.info(f"LightGBM Metrics: {lgbm_metrics}")
            
            xgb_model, _, xgb_metrics = self.train_xgboost(
                X_train_smote, y_train_smote, X_test, y_test
            )
            results['xgboost'] = xgb_metrics
            logger.info(f"XGBoost Metrics: {xgb_metrics}")
            
            # ═══════════════════════════════════════════════════════════════
            #  PHASE 4: Model selection — notebook-aligned: ROC-AUC selects best model
            # ═══════════════════════════════════════════════════════════════
            best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
            logger.info(
                f"Best model: {best_model_name} "
                f"(ROC-AUC: {results[best_model_name]['roc_auc']:.4f})"
            )
            
            # Resolve best-model object, threshold, scaler
            if best_model_name == 'logistic_regression':
                best_model = lr_model
                best_scaler = lr_scaler
                best_threshold = lr_metrics['threshold']
                if best_scaler is not None:
                    best_proba = best_model.predict_proba(best_scaler.transform(X_test))[:, 1]
                    train_proba = best_model.predict_proba(best_scaler.transform(X_train))[:, 1]
                else:
                    best_proba = best_model.predict_proba(X_test.values)[:, 1]
                    train_proba = best_model.predict_proba(X_train.values)[:, 1]
            elif best_model_name == 'random_forest':
                best_model = rf_model
                best_scaler = None
                best_threshold = rf_metrics['threshold']
                best_proba = best_model.predict_proba(X_test.values)[:, 1]
                train_proba = best_model.predict_proba(X_train.values)[:, 1]
            elif best_model_name == 'lightgbm':
                best_model = lgbm_model
                best_scaler = None
                best_threshold = lgbm_metrics['threshold']
                best_proba = best_model.predict_proba(X_test.values)[:, 1]
                train_proba = best_model.predict_proba(X_train.values)[:, 1]
            else:
                best_model = xgb_model
                best_scaler = None
                best_threshold = xgb_metrics['threshold']
                best_proba = best_model.predict_proba(X_test.values)[:, 1]
                train_proba = best_model.predict_proba(X_train.values)[:, 1]
            
            # Probability range logging
            logger.info(
                f'Probability range: {best_proba.min():.4f} to {best_proba.max():.4f}'
            )
            
            # Train-test AUC gap
            train_auc = roc_auc_score(y_train, train_proba)
            test_auc = roc_auc_score(y_test, best_proba)
            auc_gap = train_auc - test_auc
            logger.info(
                f'Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | '
                f'AUC Gap: {auc_gap:.4f}'
            )
            if auc_gap > 0.10:
                logger.warning(f'Potential overfitting detected: AUC gap = {auc_gap:.4f}')
            
            # ═══════════════════════════════════════════════════════════════
            #  PHASE 5: Persist all artifacts
            # ═══════════════════════════════════════════════════════════════
            os.makedirs(self.config.root_dir, exist_ok=True)
            os.makedirs('models', exist_ok=True)
            
            model_path = os.path.join('models', 'best_model.pkl')
            joblib.dump(best_model, model_path)
            logger.info(f"Saved best model to {model_path}")
            
            threshold_path = os.path.join('models', 'threshold.pkl')
            joblib.dump(best_threshold, threshold_path)
            logger.info(f"Saved threshold ({best_threshold:.3f}) to {threshold_path}")
            
            if best_scaler is not None:
                scaler_path = os.path.join('models', "scaler.pkl")
                joblib.dump(best_scaler, scaler_path)
                logger.info(f"Saved scaler to {scaler_path}")
            
            # Feature names (from preprocessor — the single source of truth)
            feature_names = preprocessor.feature_names_
            save_json(
                os.path.join('models', "feature_names.json"),
                {"features": feature_names, "count": len(feature_names)}
            )
            
            # Model comparison
            save_json(os.path.join(self.config.root_dir, "model_comparison.json"), results)
            comparison_df = pd.DataFrame(results).T
            comparison_df.to_csv(self.config.comparison_csv_path)
            logger.info(f'Saved model comparison CSV to {self.config.comparison_csv_path}')
            
            # Classification report
            best_pred = (best_proba >= best_threshold).astype(int)
            report = classification_report(
                y_test, best_pred,
                target_names=['Not Recovered', 'Recovered'],
                output_dict=True
            )
            save_json(self.config.classification_report_path, report)
            logger.info(f'Saved classification report to {self.config.classification_report_path}')
            
            # Plots
            self.save_feature_importance(
                best_model, feature_names, self.config.feature_importance_path
            )
            self.save_roc_curve(y_test, best_proba, self.config.roc_curve_path)
            self.save_confusion_matrix_plot(y_test, best_pred, self.config.confusion_matrix_path)
            self.save_pr_curve(y_test, best_proba, self.config.pr_curve_path)
            
            # Summary
            best_metrics = results[best_model_name]
            logger.info(
                f'Accuracy: {best_metrics["accuracy"]:.4f} | '
                f'Precision: {best_metrics["precision"]:.4f} | '
                f'Recall: {best_metrics["recall"]:.4f} | '
                f'F1: {best_metrics["f1"]:.4f} | '
                f'ROC-AUC: {best_metrics["roc_auc"]:.4f}'
            )
            
            return model_path, results[best_model_name]['roc_auc']
            
        except Exception as e:
            raise LoanRecoveryException(e, sys)
