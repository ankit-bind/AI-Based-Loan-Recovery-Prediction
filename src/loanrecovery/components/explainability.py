import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import json
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import shap
from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException

logger = get_logger(__name__)

class Explainability:
    def __init__(self, config):
        self.config = config

    def initiate_explainability(self, engineered_data_path: str, output_dir: str):
        logger.info('Loan Recovery Explainability Analysis Started')
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Load saved test set
            test_data_path = os.path.join(self.config.root_dir, "..", "model_trainer", "test_data.pkl")
            test_data_path = os.path.abspath(test_data_path)
            
            if os.path.exists(test_data_path):
                test_data = joblib.load(test_data_path)
                X_test = test_data["X_test"]
                X_sample = X_test.iloc[:min(1000, len(X_test))]
            else:
                # Fallback
                logger.warning("Saved test set not found, using engineered data")
                df = pd.read_csv(engineered_data_path)
                X = df.drop(columns=[self.config.target_column])
                test_size = min(int(len(df) * self.config.test_size), 5000)
                X_sample = X.tail(test_size).iloc[:1000]
            
            # Feature alignment with feature_names.json
            feature_path = os.path.join('models', 'feature_names.json')
            if os.path.exists(feature_path):
                with open(feature_path, 'r') as f:
                    feature_info = json.load(f)
                    expected_features = feature_info['features']
                    # Align columns
                    for col in expected_features:
                        if col not in X_sample.columns:
                            X_sample[col] = 0
                    X_sample = X_sample[expected_features]
                    logger.info('Feature alignment completed successfully')
            
            # Remaining null validation
            remaining_nulls = X_sample.isnull().sum().sum()
            logger.info(f'Remaining null values: {remaining_nulls}')
            
            # Load model
            model_path = os.path.join('models', 'best_model.pkl')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Model not found at {model_path}')
            model = joblib.load(model_path)
            
            # Check if scaler exists
            scaler_path = os.path.join('models', "scaler.pkl")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                X_sample_transformed = scaler.transform(X_sample)
                X_sample_shap = pd.DataFrame(X_sample_transformed, columns=X_sample.columns)
            else:
                X_sample_shap = X_sample
            
            logger.info(f'SHAP sample size: {X_sample_shap.shape}')
            
            # SHAP summary plot
            logger.info("Generating SHAP summary plot")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample_shap)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_values_to_plot = shap_values[1]
                else:
                    shap_values_to_plot = shap_values
                
                # Summary plot (beeswarm)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values_to_plot, X_sample_shap, show=False)
                summary_path = os.path.join(output_dir, "shap_summary.png")
                plt.savefig(summary_path, bbox_inches='tight', dpi=150)
                plt.close()
                logger.info(f"SHAP summary plot saved to {summary_path}")
                
                # Bar plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values_to_plot, X_sample_shap, plot_type='bar', show=False)
                bar_path = os.path.join(output_dir, "shap_bar_plot.png")
                plt.savefig(bar_path, bbox_inches='tight', dpi=150)
                plt.close()
                logger.info(f'SHAP bar plot saved to {bar_path}')
                
                # Waterfall plot (single sample)
                if len(X_sample_shap) > 0:
                    try:
                        plt.figure(figsize=(10, 6))
                        # For TreeExplainer, expected_value might be a list
                        expected_val = explainer.expected_value
                        if isinstance(expected_val, list):
                            expected_val = expected_val[1]
                        
                        shap.plots.waterfall(
                            shap.Explanation(
                                values=shap_values_to_plot[0],
                                base_values=expected_val,
                                data=X_sample_shap.iloc[0],
                                feature_names=X_sample_shap.columns
                            ),
                            show=False
                        )
                        waterfall_path = os.path.join(output_dir, "shap_waterfall.png")
                        plt.savefig(waterfall_path, bbox_inches='tight', dpi=150)
                        plt.close()
                        logger.info(f'SHAP waterfall plot saved to {waterfall_path}')
                    except Exception as waterfall_error:
                        logger.warning(f'SHAP waterfall plot failed: {waterfall_error}')
                        waterfall_path = None
                
                # Feature importance export
                importance_df = pd.DataFrame({
                    'feature': X_sample_shap.columns,
                    'importance': np.abs(shap_values_to_plot).mean(axis=0)
                }).sort_values(by='importance', ascending=False)
                
                importance_path = os.path.join(output_dir, 'shap_feature_importance.csv')
                importance_df.to_csv(importance_path, index=False)
                logger.info('SHAP feature importance exported successfully')
                
                return summary_path
                
            except Exception as shap_error:
                logger.warning(f"SHAP TreeExplainer failed: {shap_error}")
                return None

        except Exception as e:
            raise LoanRecoveryException(e, sys)
