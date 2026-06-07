import os
import sys
import pandas as pd
from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException
from src.loanrecovery.utils import save_json

logger = get_logger(__name__)

class DataValidation:
    def __init__(self, config):
        self.config = config

    def validate_all_columns(self, df: pd.DataFrame) -> bool:
        """Validate that all expected columns exist in the dataset."""
        try:
            # Read schema
            import yaml
            with open('config/schema.yaml', 'r') as f:
                schema = yaml.safe_load(f)

            expected_columns = schema.get("validation", {}).get("required_columns", [])
            actual_columns = list(df.columns)

            missing = [col for col in expected_columns if col not in actual_columns]

            validation_status = len(missing) == 0

            report = {
                "validation_status": validation_status,
                "total_columns": len(actual_columns),
                "expected_columns": len(expected_columns),
                "missing_columns": missing,
            }

            # Save report
            os.makedirs(self.config.root_dir, exist_ok=True)
            save_json(self.config.report_path, report)
            logger.info(f'Validation report saved at: {self.config.report_path}')

            if missing:
                logger.warning(f"Missing columns: {missing}")

            logger.info(f"Data validation status: {validation_status}")
            return validation_status

        except Exception as e:
            raise LoanRecoveryException(e, sys)

    def initiate_data_validation(self, data_path: str):
        logger.info('Loan Recovery Data Validation Started')
        try:
            # Validate Dataset File Path
            if not os.path.exists(data_path):
                logger.error(f'Dataset not found at: {data_path}')
                return False

            # Read train data
            df = pd.read_csv(data_path)
            logger.info(f"Dataset loaded for validation with shape: {df.shape}")

            # Add Empty Dataset Validation
            if df.empty:
                logger.error('Dataset is empty')
                return False

            # Validate Target Column Existence
            if self.config.target_column not in df.columns:
                logger.error(f'Target column {self.config.target_column} not found')
                return False

            # Validate Null Values in Target Column
            target_nulls = df[self.config.target_column].isnull().sum()
            if target_nulls > 0:
                logger.error(f'Target column contains {target_nulls} null values')
                return False

            # Check minimum rows
            if hasattr(self.config, 'min_rows') and len(df) < self.config.min_rows:
                logger.error(f"Dataset has {len(df)} rows, minimum required: {self.config.min_rows}")
                return False

            # Validate Minimum Feature Count
            if df.shape[1] < 20:
                logger.error('Insufficient number of features')
                return False

            # Validate Class Distribution
            class_distribution = df[self.config.target_column].value_counts()
            logger.info(f'Class distribution: {class_distribution.to_dict()}')
            if len(class_distribution) < 2:
                logger.error('Only one target class found')
                return False

            # Add Duplicate Row Validation
            duplicate_count = df.duplicated().sum()
            logger.info(f'Duplicate rows found: {duplicate_count}')

            # Add Remaining Null Value Check
            remaining_nulls = df.isnull().sum().sum()
            logger.info(f'Remaining null values: {remaining_nulls}')

            # Validate columns
            status = self.validate_all_columns(df)

            if not status:
                logger.error("Data validation failed: missing expected columns")
            else:
                logger.info('Loan Recovery Data Validation Passed Successfully')

            return status

        except Exception as e:
            raise LoanRecoveryException(e, sys)
