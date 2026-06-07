"""
Unified Preprocessing Pipeline for Loan Recovery Prediction
============================================================

This module provides a single, serializable preprocessor class that
encapsulates ALL data cleaning, feature engineering, encoding, imputation,
and alignment logic.  It is designed to be:

    1. fit()   once on the training data
    2. saved   alongside the model artifact
    3. transform() every incoming inference row with *exactly* the same logic

By centralising preprocessing in one class we eliminate the
training-inference skew that previously caused:
    - missing engineered features
    - categorical encoding mismatches
    - silent zero-filling of entire columns
    - SHAP instability
"""

import os
import sys
import re
import json
import joblib
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Constants ──────────────────────────────────────────────────────────────

BINARY_MAP = {
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "Y",
    "FLAG_OWN_REALTY": "Y",
    "NAME_CONTRACT_TYPE": "Cash loans",
}

ONEHOT_COLS = ["NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
               "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"]

FREQ_COLS = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]

# Columns that the *training* pipeline drops because they are raw aggregates
# or highly correlated.  We keep them out of the final feature matrix.
DROP_COLS = [
    "SK_ID_CURR",
    "bureau_total_loans", "bureau_active_loans", "bureau_closed_loans",
    "bureau_total_credit", "bureau_total_debt",
    "bureau_total_overdue", "bureau_total_prolonged",
    "FLAG_EMP_PHONE", "NAME_INCOME_TYPE_Pensioner",
]

# Document flags that are aggregated into TOTAL_DOCUMENTS_SUBMITTED
DOC_FLAGS = [f"FLAG_DOCUMENT_{i}" for i in
             [3, 5, 6, 8, 9, 11, 13, 14, 15, 16, 18, 19, 20, 21]]

INQUIRY_COLS = [
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
]

SOCIAL_COLS = [
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
]

ZERO_FILL_COLS = [
    "DAYS_EMPLOYED", "EMPLOYMENT_YEARS", "EMPLOYMENT_AGE_RATIO",
    "TOTAL_SOCIAL_DEFAULTS", "TOTAL_SOCIAL_OBS", "EXT_SOURCE_STD",
    "OCCUPATION_TYPE_FREQ",
]

MEDIAN_FILL_COLS = [
    "CNT_FAM_MEMBERS", "ANNUITY_INCOME_RATIO", "GOODS_CREDIT_RATIO",
    "INCOME_PER_PERSON", "CHILDREN_RATIO", "ACTIVE_LOAN_RATIO",
    "CLOSED_LOAN_RATIO", "OVERDUE_PER_LOAN", "PROLONGED_LOAN_RATIO",
    "CREDIT_UTILIZATION_RATIO",
]


class UnifiedPreprocessor(BaseEstimator, TransformerMixin):
    """
    End-to-end preprocessor for the Loan Recovery pipeline.

    Parameters
    ----------
    target_column : str
        Name of the target column (excluded from transformation).
    apply_scaler : bool
        Whether to fit a StandardScaler on numeric features.  Only needed
        for Logistic Regression; tree-based models ignore it.
    """

    def __init__(self, target_column: str = "TARGET", apply_scaler: bool = False):
        self.target_column = target_column
        self.apply_scaler = apply_scaler

        # ── Fitted attributes (populated during fit()) ──
        self.feature_names_: Optional[List[str]] = None
        self.onehot_categories_: Dict[str, List[str]] = {}
        self.freq_maps_: Dict[str, Dict] = {}
        self.defaults_: Dict[str, float] = {}
        self.scaler_: Optional[StandardScaler] = None
        self.dtypes_: Dict[str, str] = {}

    # ═══════════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════

    def fit(self, df: pd.DataFrame) -> "UnifiedPreprocessor":
        """
        Learn all preprocessing parameters from the training DataFrame.
        """
        logger.info("UnifiedPreprocessor.fit() started")
        df = df.copy()

        # 1. Learn defaults BEFORE any feature engineering so that inference
        #    rows that are missing bureau columns still get sensible values.
        self._learn_defaults(df)

        # 2. Run the full transformation pipeline on training data
        df_transformed = self._transform_core(df, fit_mode=True)

        # 3. Store final feature order (excluding target)
        self.feature_names_ = [c for c in df_transformed.columns
                               if c != self.target_column]
        self.dtypes_ = {c: str(df_transformed[c].dtype) for c in self.feature_names_}

        # 4. Fit scaler if requested
        if self.apply_scaler:
            numeric = df_transformed[self.feature_names_].select_dtypes(include=[np.number])
            self.scaler_ = StandardScaler()
            self.scaler_.fit(numeric.values)
            logger.info(f"StandardScaler fitted on {numeric.shape[1]} numeric features")

        logger.info(f"UnifiedPreprocessor fitted. Final features: {len(self.feature_names_)}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the learnt preprocessing to new data (inference).

        Returns a DataFrame with EXACTLY the same columns and order as
        the training matrix that the model was fit on.
        """
        if self.feature_names_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit() first.")

        df = df.copy()
        df_transformed = self._transform_core(df, fit_mode=False)

        # ── Final alignment: add missing / drop extra / reorder ──
        df_transformed = self._align_features(df_transformed)

        # ── Dtype enforcement ──
        for col, dtype in self.dtypes_.items():
            if col in df_transformed.columns:
                try:
                    if "int" in dtype:
                        df_transformed[col] = df_transformed[col].astype(dtype)
                    elif "float" in dtype:
                        df_transformed[col] = df_transformed[col].astype(dtype)
                except (ValueError, TypeError):
                    pass

        logger.info(f"Preprocessor.transform() complete: shape={df_transformed.shape}")
        return df_transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience wrapper. Preserves the target column in the output
        so downstream splitting still works."""
        self.fit(df)
        result = self.transform(df)
        # Preserve target if it was in the input but got dropped by alignment
        if self.target_column in df.columns and self.target_column not in result.columns:
            result[self.target_column] = df[self.target_column].values
        return result

    # ═══════════════════════════════════════════════════════════════════════
    #  SERIALISATION
    # ═══════════════════════════════════════════════════════════════════════

    def save(self, path: str):
        """Save the fitted preprocessor with joblib."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"UnifiedPreprocessor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "UnifiedPreprocessor":
        """Load a previously saved preprocessor."""
        obj = joblib.load(path)
        logger.info(f"UnifiedPreprocessor loaded from {path}")
        return obj

    # ═══════════════════════════════════════════════════════════════════════
    #  INTERNAL PIPELINE
    # ═══════════════════════════════════════════════════════════════════════

    def _transform_core(self, df: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        """Run every preprocessing step.  When fit_mode=True we *learn*
        parameters (freq maps, one-hot categories); otherwise we *apply*
        the learnt parameters."""

        df = df.copy()

        # 1. Data cleaning (identical for train & inference)
        df = self._clean_data(df)

        # 2. Feature engineering (ratios, aggregates, etc.)
        df = self._engineer_features(df)

        # 3. Binary encoding
        df = self._binary_encode(df)

        # 4. Frequency encoding (learned during fit, applied during inference)
        df = self._frequency_encode(df, fit_mode=fit_mode)

        # 5. One-hot encoding (learned during fit, applied during inference)
        df = self._onehot_encode(df, fit_mode=fit_mode)

        # 6. Missing-value imputation using learnt defaults
        df = self._impute_missing(df)

        return df

    # ── Step 1: Data Cleaning ─────────────────────────────────────────────

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replicate notebook cleaning steps.  Idempotent — safe to run
        on already-cleaned data."""
        # Drop low-info document columns
        for col in ["FLAG_DOCUMENT_2", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_7",
                    "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_17"]:
            df = df.drop(columns=[col], errors="ignore")

        df = df.drop(columns=["FLAG_MOBIL"], errors="ignore")

        # Employment anomaly flag
        if "DAYS_EMPLOYED" in df.columns:
            # Only create flag if column still contains raw sentinel value
            if (df["DAYS_EMPLOYED"] == 365243).any():
                df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
                df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
            elif "DAYS_EMPLOYED_ANOMALY" not in df.columns:
                # Already cleaned — create flag as all-zeros
                df["DAYS_EMPLOYED_ANOMALY"] = 0

        # Gender XNA fix
        if "CODE_GENDER" in df.columns:
            if df["CODE_GENDER"].dtype.kind in "OSUb" or df["CODE_GENDER"].astype(str).str.contains("XNA").any():
                df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", "F")

        # OWN_CAR_AGE conditional imputation
        if "OWN_CAR_AGE" in df.columns and "FLAG_OWN_CAR" in df.columns:
            if df["FLAG_OWN_CAR"].dtype.kind in "OSUb":
                car_n = df["FLAG_OWN_CAR"] == "N"
                car_y = df["FLAG_OWN_CAR"] == "Y"
            else:
                car_n = df["FLAG_OWN_CAR"] == 0
                car_y = df["FLAG_OWN_CAR"] == 1
            df.loc[car_n, "OWN_CAR_AGE"] = df.loc[car_n, "OWN_CAR_AGE"].fillna(0)
            median_age = df.loc[car_y, "OWN_CAR_AGE"].median()
            if pd.notna(median_age):
                df.loc[car_y, "OWN_CAR_AGE"] = df.loc[car_y, "OWN_CAR_AGE"].fillna(median_age)
            df["OWN_CAR_AGE"] = df["OWN_CAR_AGE"].fillna(0)

        return df

    # ── Step 2: Feature Engineering ───────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create every engineered feature used by the model.
        Idempotent — skips features that already exist."""

        # Financial ratios
        if "CREDIT_INCOME_RATIO" not in df.columns and all(c in df.columns for c in ["AMT_CREDIT", "AMT_INCOME_TOTAL"]):
            df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
        if "ANNUITY_INCOME_RATIO" not in df.columns and all(c in df.columns for c in ["AMT_ANNUITY", "AMT_INCOME_TOTAL"]):
            df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
        if "GOODS_CREDIT_RATIO" not in df.columns and all(c in df.columns for c in ["AMT_GOODS_PRICE", "AMT_CREDIT"]):
            df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1)
        if "CREDIT_ANNUITY_RATIO" not in df.columns and all(c in df.columns for c in ["AMT_CREDIT", "AMT_ANNUITY"]):
            df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)

        # Family features
        if "INCOME_PER_PERSON" not in df.columns and all(c in df.columns for c in ["AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"]):
            df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
        if "CHILDREN_RATIO" not in df.columns and all(c in df.columns for c in ["CNT_CHILDREN", "CNT_FAM_MEMBERS"]):
            df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / (df["CNT_FAM_MEMBERS"] + 1)

        # External credit features
        ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
        if ext_cols and "EXT_SOURCE_MEAN" not in df.columns:
            df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        if ext_cols and "EXT_SOURCE_STD" not in df.columns:
            df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
        if "EXT_SOURCE_1" in df.columns and "EXT_SOURCE_1_MISSING" not in df.columns:
            df["EXT_SOURCE_1_MISSING"] = df["EXT_SOURCE_1"].isnull().astype(int)

        # Collateral
        if "HAS_COLLATERAL" not in df.columns and all(c in df.columns for c in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]):
            # Handle both string and already-encoded binary
            car_vals = df["FLAG_OWN_CAR"] if df["FLAG_OWN_CAR"].dtype.kind in "OSUb" else df["FLAG_OWN_CAR"].astype(str)
            realty_vals = df["FLAG_OWN_REALTY"] if df["FLAG_OWN_REALTY"].dtype.kind in "OSUb" else df["FLAG_OWN_REALTY"].astype(str)
            df["HAS_COLLATERAL"] = ((car_vals == "Y") | (realty_vals == "Y")).astype(int)

        # Bureau-derived features (if bureau aggregates are present)
        bureau_count = "SK_ID_BUREAU_count"
        if bureau_count in df.columns:
            if "BUREAU_DEBT_RATIO" not in df.columns:
                df["BUREAU_DEBT_RATIO"] = df.get("AMT_CREDIT_SUM_DEBT_sum", 0) / (df.get("AMT_CREDIT_SUM_sum", 0) + 1)
            if "AVG_CREDIT_PER_LOAN" not in df.columns:
                df["AVG_CREDIT_PER_LOAN"] = df.get("AMT_CREDIT_SUM_sum", 0) / (df[bureau_count] + 1)
            if "DEBT_PER_LOAN" not in df.columns:
                df["DEBT_PER_LOAN"] = df.get("AMT_CREDIT_SUM_DEBT_sum", 0) / (df[bureau_count] + 1)

            # Ensure raw bureau count columns exist
            for raw_col in ["bureau_active_loans", "bureau_closed_loans",
                            "bureau_total_loans", "bureau_total_overdue",
                            "bureau_total_prolonged"]:
                if raw_col not in df.columns:
                    df[raw_col] = 0

            if "ACTIVE_LOAN_RATIO" not in df.columns:
                df["ACTIVE_LOAN_RATIO"] = df["bureau_active_loans"] / (df["bureau_total_loans"] + 1)
            if "CLOSED_LOAN_RATIO" not in df.columns:
                df["CLOSED_LOAN_RATIO"] = df["bureau_closed_loans"] / (df["bureau_total_loans"] + 1)
            if "OVERDUE_PER_LOAN" not in df.columns:
                df["OVERDUE_PER_LOAN"] = df["bureau_total_overdue"] / (df["bureau_total_loans"] + 1)
            if "PROLONGED_LOAN_RATIO" not in df.columns:
                df["PROLONGED_LOAN_RATIO"] = df["bureau_total_prolonged"] / (df["bureau_total_loans"] + 1)
            if "CREDIT_UTILIZATION_RATIO" not in df.columns:
                df["CREDIT_UTILIZATION_RATIO"] = df.get("bureau_total_debt", 0) / (df.get("bureau_total_credit", 0) + 1)

            # Drop intermediate raw bureau aggregates so they do not leak into the final feature set
            intermediate_bureau = [
                "bureau_active_loans", "bureau_closed_loans",
                "bureau_total_loans", "bureau_total_overdue",
                "bureau_total_prolonged"
            ]
            df = df.drop(columns=[c for c in intermediate_bureau if c in df.columns], errors="ignore")

        # Age / employment
        if "AGE_YEARS" not in df.columns and "DAYS_BIRTH" in df.columns:
            df["AGE_YEARS"] = np.abs(df["DAYS_BIRTH"]) / 365
        if "EMPLOYMENT_YEARS" not in df.columns and "DAYS_EMPLOYED" in df.columns:
            df["EMPLOYMENT_YEARS"] = np.abs(df["DAYS_EMPLOYED"]) / 365
        if "EMPLOYMENT_AGE_RATIO" not in df.columns and all(c in df.columns for c in ["EMPLOYMENT_YEARS", "AGE_YEARS"]):
            df["EMPLOYMENT_AGE_RATIO"] = df["EMPLOYMENT_YEARS"] / (df["AGE_YEARS"] + 1)

        # Document aggregate
        available_docs = [c for c in DOC_FLAGS if c in df.columns]
        if available_docs and "TOTAL_DOCUMENTS_SUBMITTED" not in df.columns:
            df["TOTAL_DOCUMENTS_SUBMITTED"] = df[available_docs].sum(axis=1)

        # Inquiry aggregate
        available_inq = [c for c in INQUIRY_COLS if c in df.columns]
        if available_inq and "TOTAL_INQUIRIES" not in df.columns:
            df["TOTAL_INQUIRIES"] = df[available_inq].sum(axis=1)
            recent = [c for c in ["AMT_REQ_CREDIT_BUREAU_DAY",
                                    "AMT_REQ_CREDIT_BUREAU_WEEK",
                                    "AMT_REQ_CREDIT_BUREAU_MON"] if c in df.columns]
            if recent and "RECENT_INQUIRY_RATIO" not in df.columns:
                df["RECENT_INQUIRY_RATIO"] = df[recent].sum(axis=1) / (df["TOTAL_INQUIRIES"] + 1)

        # Social circle aggregate
        available_soc = [c for c in SOCIAL_COLS if c in df.columns]
        if available_soc and "TOTAL_SOCIAL_DEFAULTS" not in df.columns:
            df["TOTAL_SOCIAL_DEFAULTS"] = df.get("DEF_30_CNT_SOCIAL_CIRCLE", 0) + df.get("DEF_60_CNT_SOCIAL_CIRCLE", 0)
        if available_soc and "TOTAL_SOCIAL_OBS" not in df.columns:
            df["TOTAL_SOCIAL_OBS"] = df.get("OBS_30_CNT_SOCIAL_CIRCLE", 0) + df.get("OBS_60_CNT_SOCIAL_CIRCLE", 0)

        return df

    # ── Step 3: Binary Encoding ───────────────────────────────────────────

    def _binary_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary encoding — idempotent (safe on already-encoded ints)."""
        for col, true_val in BINARY_MAP.items():
            if col not in df.columns:
                continue
            # Skip if already numeric (0/1)
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            df[col] = (df[col].astype(str).str.strip() == true_val).astype(int)
        return df

    # ── Step 4: Frequency Encoding ──────────────────────────────────────────

    def _frequency_encode(self, df: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        """Frequency encoding — idempotent (skips if _FREQ already exists)."""
        for col in FREQ_COLS:
            if col not in df.columns:
                continue
            freq_col = f"{col}_FREQ"
            if freq_col in df.columns:
                # Already encoded — drop source if it still exists
                df = df.drop(columns=[col], errors="ignore")
                continue

            if fit_mode:
                freq_map = df[col].value_counts().to_dict()
                self.freq_maps_[col] = freq_map
            else:
                freq_map = self.freq_maps_.get(col, {})

            df[freq_col] = df[col].map(freq_map).fillna(0)
            df = df.drop(columns=[col], errors="ignore")

        return df

    # ── Step 5: One-Hot Encoding ────────────────────────────────────────────

    def _onehot_encode(self, df: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        """One-hot encoding — idempotent (skips if source columns are missing
        or if OHE columns already exist)."""
        available = [c for c in ONEHOT_COLS if c in df.columns]
        if not available:
            return df

        if fit_mode:
            for col in available:
                self.onehot_categories_[col] = sorted(df[col].dropna().unique().tolist())

        # Use pandas get_dummies
        df_ohe = pd.get_dummies(df, columns=available, drop_first=True, dtype=int)

        # Sanitise column names (match training pipeline)
        clean_map = {}
        for c in df_ohe.columns:
            clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(c))
            clean = re.sub(r"_+", "_", clean).rstrip("_")
            clean_map[c] = clean
        df_ohe.rename(columns=clean_map, inplace=True)

        return df_ohe

    # ── Step 6: Missing-Value Imputation ──────────────────────────────────

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using the defaults learnt during fit()."""
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            default = self.defaults_.get(col, 0)
            df[col] = df[col].fillna(default)
        return df

    # ── Final Alignment ─────────────────────────────────────────────────────

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has exactly the columns the model expects,
        in the exact order, with sensible defaults for anything missing."""

        # Add missing columns with default values
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = self.defaults_.get(col, 0)

        # Drop any extra columns the model doesn't know about
        extra_cols = [c for c in df.columns if c not in self.feature_names_]
        if extra_cols:
            logger.info(f"Dropping {len(extra_cols)} extra columns not in training feature list")
            df = df.drop(columns=extra_cols)

        # Enforce exact column order
        df = df[self.feature_names_]
        return df

    # ── Default Learning ────────────────────────────────────────────────────

    def _learn_defaults(self, df: pd.DataFrame):
        """Compute a default value for every column that appears during fit.
        Numeric columns → median
        Categorical / binary columns → mode (or 0 if all missing)"""
        self.defaults_ = {}

        for col in df.columns:
            if col == self.target_column:
                continue

            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                self.defaults_[col] = series.median() if not series.isnull().all() else 0.0
            else:
                mode_val = series.mode()
                self.defaults_[col] = mode_val[0] if len(mode_val) > 0 else 0

        logger.info(f"Learned default values for {len(self.defaults_)} columns")

    # ═══════════════════════════════════════════════════════════════════════
    #  SCALER WRAPPER (optional)
    # ═══════════════════════════════════════════════════════════════════════

    def transform_and_scale(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience method: transform() then apply StandardScaler."""
        df_t = self.transform(df)
        if self.scaler_ is None:
            return df_t.values
        numeric = df_t.select_dtypes(include=[np.number])
        return self.scaler_.transform(numeric.values)

    def scale_features(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted scaler to a numeric matrix."""
        if self.scaler_ is None:
            return X
        return self.scaler_.transform(X)
