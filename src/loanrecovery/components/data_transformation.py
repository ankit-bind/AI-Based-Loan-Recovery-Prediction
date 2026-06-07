import os
import sys
import time
import json
import pandas as pd
import numpy as np
from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException

logger = get_logger(__name__)

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def _safe_save_csv(self, df, path, max_retries=3):
        """Save DataFrame to CSV with retry logic for Windows file locks."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for attempt in range(max_retries):
            try:
                tmp_path = path + ".tmp"
                df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, path)
                return True
            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"File lock on {path}, retrying in 1s... ({e})")
                    time.sleep(1)
                else:
                    ts = int(time.time())
                    fallback = path.replace(".csv", f"_{ts}.csv")
                    df.to_csv(fallback, index=False)
                    logger.warning(f"Saved to fallback path: {fallback}")
                    return fallback
        return False

    def initiate_data_transformation(self, engineered_data_path: str):
        logger.info('Loan Recovery Data Transformation Started')
        try:
            df = pd.read_csv(engineered_data_path)
            logger.info(f"Loaded engineered data: {df.shape}")
            
            # Target Column Validation
            if self.config.target_column not in df.columns:
                raise ValueError(f'Target column {self.config.target_column} not found')

            # Target Null Check
            target_nulls = df[self.config.target_column].isnull().sum()
            if target_nulls > 0:
                raise ValueError('Target column contains null values')

            # Duplicate Row Handling
            duplicate_count = df.duplicated().sum()
            logger.info(f'Duplicate rows found: {duplicate_count}')
            if duplicate_count > 0:
                df = df.drop_duplicates()
                logger.info('Duplicate rows removed successfully')

            # Check for any remaining missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                logger.warning(f"Found {missing_count} missing values, filling with median")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df[col].isnull().sum() > 0:
                        df[col] = df[col].fillna(df[col].median())
            
            # Save transformed data
            output_dir = self.config.root_dir if hasattr(self.config, 'root_dir') else 'artifacts/data_transformation'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "transformed_data.csv")
            result = self._safe_save_csv(df, output_path)
            if result is True:
                logger.info(f'Transformed dataset saved at: {output_path}')
            else:
                logger.warning(f'Saved to fallback: {result}')
                output_path = result if isinstance(result, str) else output_path

            # Save Feature Names
            feature_names = list(df.columns)
            if self.config.target_column in feature_names:
                feature_names.remove(self.config.target_column)

            feature_path = os.path.join(output_dir, 'feature_names.json')
            with open(feature_path, 'w') as f:
                json.dump(feature_names, f)
            logger.info('Feature names saved successfully')
            
            logger.info('Loan Recovery Data Transformation Completed Successfully')
            return output_path
            
        except Exception as e:
            raise LoanRecoveryException(e, sys)
