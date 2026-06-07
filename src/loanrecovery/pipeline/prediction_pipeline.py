import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException
from src.loanrecovery.pipeline.unified_preprocessor import UnifiedPreprocessor

logger = get_logger(__name__)


class PredictPipeline:
    """
    Production prediction pipeline.

    Loads the fitted UnifiedPreprocessor alongside the model so that
    raw business inputs are transformed with *exactly* the same logic
    used during training.
    """

    def __init__(self):
        try:
            model_path = Path("models/best_model.pkl")
            threshold_path = Path("models/threshold.pkl")
            scaler_path = Path("models/scaler.pkl")
            preprocessor_path = Path("models/preprocessor.pkl")

            # ── Validate artifact existence ──
            for p, name in [(model_path, "Model"),
                            (threshold_path, "Threshold"),
                            (preprocessor_path, "Preprocessor")]:
                if not p.exists():
                    raise FileNotFoundError(
                        f"{name} artifact not found at {p}. "
                        f"Please run the training pipeline first."
                    )

            # ── Load model ──
            try:
                self.model = joblib.load(model_path)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"{e}\n\n"
                    "LightGBM not found in current Python environment.\n"
                    "Please run Streamlit through your virtual environment:\n"
                    "  myenv\\Scripts\\python.exe -m streamlit run app.py\n"
                    "Do NOT use: streamlit run app.py"
                ) from e

            # ── Load threshold ──
            self.threshold = joblib.load(threshold_path)
            if not isinstance(self.threshold, (float, np.floating)):
                self.threshold = float(self.threshold)

            # ── Load optional scaler (Logistic Regression only) ──
            self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None

            # ── Load unified preprocessor (the critical piece) ──
            self.preprocessor = UnifiedPreprocessor.load(preprocessor_path)
            self.feature_names = self.preprocessor.feature_names_

            logger.info(
                f"PredictPipeline loaded successfully | "
                f"Features: {len(self.feature_names)} | "
                f"Threshold: {self.threshold:.4f}"
            )

        except Exception as e:
            raise LoanRecoveryException(e, sys)

    # ═══════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════

    def predict(self, raw_features: pd.DataFrame):
        """
        Predict a single borrower.

        Parameters
        ----------
        raw_features : pd.DataFrame
            Raw business inputs (as collected from the Streamlit UI).

        Returns
        -------
        prediction : int
        probability : float
        risk_label : str
        aligned_features : pd.DataFrame
            The fully transformed feature matrix (useful for SHAP / debugging).
        """
        try:
            # 1. Unified preprocessing (clean → engineer → encode → align)
            transformed = self.preprocessor.transform(raw_features)
            logger.info(
                f"Preprocessing complete | Input cols: {raw_features.shape[1]} | "
                f"Output cols: {transformed.shape[1]}"
            )

            # 2. Validate feature count matches expectation
            expected = len(self.feature_names)
            actual = transformed.shape[1]
            if actual != expected:
                raise ValueError(
                    f"Feature count mismatch after preprocessing: "
                    f"expected {expected}, got {actual}"
                )

            # 3. Optional scaling (Logistic Regression)
            if self.scaler is not None:
                numeric = transformed.select_dtypes(include=[np.number])
                X = self.scaler.transform(numeric.values)
            else:
                X = transformed.values

            # 4. Predict probability BEFORE threshold application
            proba = self.model.predict_proba(X)[:, 1]
            logger.info(f"Raw predict_proba: {proba[0]:.4f}")

            # 5. Apply business threshold
            prediction = (proba >= self.threshold).astype(int)
            logger.info(
                f"Threshold: {self.threshold:.4f} | Prediction: {prediction[0]}"
            )

            # 6. Risk label
            risk_label = "Recovered" if prediction[0] == 1 else "Not Recovered"

            return prediction[0], float(proba[0]), risk_label, transformed

        except Exception as e:
            raise LoanRecoveryException(e, sys)

    def predict_batch(self, csv_path: str):
        """
        Batch prediction from a CSV file.

        The CSV should contain the SAME raw columns that the Streamlit UI
        collects.  The UnifiedPreprocessor handles all feature engineering
        and alignment automatically.
        """
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found at {csv_path}")

            df = pd.read_csv(csv_path)
            if df.empty:
                raise ValueError("Uploaded batch file is empty")

            logger.info(f"Batch prediction started: {len(df)} rows")

            # Unified preprocessing for the entire batch
            transformed = self.preprocessor.transform(df)
            logger.info(f"Batch preprocessing complete: {transformed.shape}")

            # Scale if needed
            if self.scaler is not None:
                numeric = transformed.select_dtypes(include=[np.number])
                X = self.scaler.transform(numeric.values)
            else:
                X = transformed.values

            # Predict probabilities BEFORE threshold
            proba = self.model.predict_proba(X)[:, 1]
            logger.info(
                f"Batch probability range: {proba.min():.4f} to {proba.max():.4f}"
            )

            # Apply threshold
            prediction = (proba >= self.threshold).astype(int)
            risk_labels = [
                "Recovered" if p == 1 else "Not Recovered" for p in prediction
            ]

            # Build results DataFrame
            df["PREDICTION"] = prediction
            df["PROBABILITY"] = proba
            df["RISK_LABEL"] = risk_labels
            df["THRESHOLD_USED"] = self.threshold
            df["PREDICTION_TIME"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            logger.info(f"Batch prediction complete: {len(df)} rows processed")
            return df

        except Exception as e:
            raise LoanRecoveryException(e, sys)
