import os
import sys
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def _safe_save_csv(self, df, path, max_retries=3):
        """Save DataFrame to CSV with retry logic for Windows file locks."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for attempt in range(max_retries):
            try:
                # Try writing to a temp file first, then atomic replace
                tmp_path = path + ".tmp"
                df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, path)
                return True
            except (OSError, PermissionError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"File lock on {path}, retrying in 1s... ({e})")
                    time.sleep(1)
                else:
                    # Fallback: use timestamped filename
                    ts = int(time.time())
                    fallback = path.replace(".csv", f"_{ts}.csv")
                    df.to_csv(fallback, index=False)
                    logger.warning(f"Saved to fallback path: {fallback}")
                    return fallback
        return False

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion")
        try:
            if not os.path.exists(self.config.application_train):
                raise FileNotFoundError(
                    f"Raw application train dataset file not found at: {self.config.application_train}"
                )

            df = pd.read_csv(self.config.application_train)
            logger.info(f"Dataset loaded with shape: {df.shape}")

            # Optional sampling
            if hasattr(self.config, 'n_samples') and self.config.n_samples and self.config.n_samples < len(df):
                df = df.sample(n=self.config.n_samples, random_state=self.config.random_state)
                logger.info(f"Sampled to {self.config.n_samples} rows")

            # Save raw snapshot (non-critical — continue even if locked)
            result = self._safe_save_csv(df, self.config.raw_data_path)
            if result is True:
                logger.info(f"Saved raw data snapshot to: {self.config.raw_data_path}")
            else:
                logger.warning(f"Could not save raw snapshot to {self.config.raw_data_path}")

            # Split
            train_set, test_set = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df[self.config.target_column]
            )
            logger.info(f"Train shape: {train_set.shape}, Test shape: {test_set.shape}")

            # Save train / test
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            self._safe_save_csv(train_set, self.config.train_data_path)
            self._safe_save_csv(test_set, self.config.test_data_path)

            logger.info(f"Data ingestion complete. Files saved to {os.path.dirname(self.config.train_data_path)}")
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise LoanRecoveryException(e, sys)
