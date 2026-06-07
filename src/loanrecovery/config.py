# ============================================================
# LOAN RECOVERY PROJECT — CONFIGURATION MANAGER
# ============================================================
"""
Centralized configuration management using dataclasses.

Loads YAML configs and exposes typed configuration objects
to all pipeline components.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException
from src.loanrecovery.utils import read_yaml

logger = get_logger(__name__)

# ── Config file paths ────────────────────────────────────────
CONFIG_FILE  = "config/config.yaml"
PARAMS_FILE  = "config/params.yaml"
SCHEMA_FILE  = "config/schema.yaml"


# ── Dataclasses ──────────────────────────────────────────────
@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion component."""
    raw_dir           : str
    processed_dir     : str
    application_train : str
    bureau            : str
    previous_app      : str
    installments      : str
    pos_cash          : str
    train_data_path   : str
    test_data_path    : str
    n_samples         : int
    random_state      : int
    test_size         : float
    chunk_size        : int
    raw_data_path     : str
    target_column     : str
    root_dir          : str


@dataclass
class DataValidationConfig:
    """Configuration for data validation component."""
    root_dir          : str
    report_path       : str
    status_path       : str
    required_columns  : list
    min_rows          : int
    target_column     : str


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation component."""
    root_dir          : str
    preprocessor_path : str
    train_arr_path    : str
    test_arr_path     : str
    target_column     : str


@dataclass
class ModelTrainerConfig:
    """Configuration for model training component."""
    root_dir          : str
    trained_model_path: str
    metrics_path      : str
    comparison_path   : str
    comparison_csv_path: str
    classification_report_path: str
    feature_importance_path: str
    roc_curve_path    : str
    confusion_matrix_path: str
    pr_curve_path     : str
    random_state      : int
    primary_metric    : str
    target_column     : str
    test_size         : float


@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation component."""
    root_dir          : str
    report_path       : str
    roc_curve_path    : str
    cm_path           : str
    pr_curve_path     : str
    feature_imp_path  : str
    threshold_high    : float
    threshold_med     : float
    target_column     : str


@dataclass
class ModelSaveConfig:
    """Configuration for model artifacts."""
    root_dir          : str
    best_model_path   : str
    preprocessor_path : str
    threshold_path    : str
    feature_names_path: str
    all_models_path   : str


# ── Configuration Manager ────────────────────────────────────
class ConfigurationManager:
    """
    Central configuration manager.

    Loads all YAML configs and exposes
    typed dataclass configurations to
    pipeline components.

    Usage:
        config = ConfigurationManager()
        ingestion_config = config.get_data_ingestion_config()
    """

    def __init__(
        self,
        config_path : str = CONFIG_FILE,
        params_path : str = PARAMS_FILE,
        schema_path : str = SCHEMA_FILE
    ) -> None:
        """
        Initialize configuration manager.

        Args:
            config_path: Path to config.yaml
            params_path: Path to params.yaml
            schema_path: Path to schema.yaml
        """
        try:
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)
            self.schema = read_yaml(schema_path)

            # Create all artifact directories
            self._create_directories()

            logger.info("ConfigurationManager initialized")

        except Exception as e:
            raise LoanRecoveryException(e, sys)

    def _create_directories(self) -> None:
        """Create all required directories."""
        dirs = [
            self.config['data']['raw_dir'],
            self.config['data']['processed_dir'],
            self.config['artifacts']['data_ingestion']['root_dir'],
            self.config['artifacts']['data_validation']['root_dir'],
            self.config['artifacts']['data_transformation']['root_dir'],
            self.config['artifacts']['model_trainer']['root_dir'],
            self.config['artifacts']['model_evaluation']['root_dir'],
            self.config['models']['root_dir'],
            self.config['logging']['log_dir'],
            self.config['reports']['root_dir'],
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        logger.info("All directories created")

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration."""
        try:
            cfg = self.config['data']
            art = self.config['artifacts']['data_ingestion']
            prm = self.params['data']

            return DataIngestionConfig(
                raw_dir           = cfg['raw_dir'],
                processed_dir     = cfg['processed_dir'],
                application_train = cfg['files']['application_train'],
                bureau            = cfg['files']['bureau'],
                previous_app      = cfg['files']['previous_application'],
                installments      = cfg['files']['installments'],
                pos_cash          = cfg['files']['pos_cash'],
                train_data_path   = art['train_data'],
                test_data_path    = art['test_data'],
                n_samples         = prm['n_samples'],
                random_state      = prm['random_state'],
                test_size         = prm['test_size'],
                chunk_size        = prm['chunk_size'],
                raw_data_path     = art['raw_data'],
                target_column     = self.schema['target_column'],
                root_dir          = art['root_dir'],
            )
        except Exception as e:
            raise LoanRecoveryException(e, sys)

    def get_data_validation_config(self) -> DataValidationConfig:
        """Get data validation configuration."""
        try:
            art = self.config['artifacts']['data_validation']
            sch = self.schema

            return DataValidationConfig(
                root_dir         = art['root_dir'],
                report_path      = art['report'],
                status_path      = art['status'],
                required_columns = sch['validation']['required_columns'],
                min_rows         = sch['validation']['min_rows'],
                target_column    = sch['target_column'],
            )
        except Exception as e:
            raise LoanRecoveryException(e, sys)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Get data transformation configuration."""
        try:
            art = self.config['artifacts']['data_transformation']

            return DataTransformationConfig(
                root_dir          = art['root_dir'],
                preprocessor_path = art['preprocessor'],
                train_arr_path    = art['train_arr'],
                test_arr_path     = art['test_arr'],
                target_column     = self.schema['target_column'],
            )
        except Exception as e:
            raise LoanRecoveryException(e, sys)

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Get model trainer configuration."""
        try:
            art = self.config['artifacts']['model_trainer']
            prm = self.params

            return ModelTrainerConfig(
                root_dir           = art['root_dir'],
                trained_model_path = art['trained_model'],
                metrics_path       = art['metrics'],
                comparison_path    = art['comparison'],
                comparison_csv_path= art['comparison_csv'],
                classification_report_path = art['classification_report'],
                feature_importance_path = art['feature_importance'],
                roc_curve_path     = art['roc_curve'],
                confusion_matrix_path = art['confusion_matrix'],
                pr_curve_path      = art['pr_curve'],
                random_state       = prm['data']['random_state'],
                primary_metric     = prm['evaluation']['primary_metric'],
                target_column      = self.schema['target_column'],
                test_size          = prm['data']['test_size'],
            )
        except Exception as e:
            raise LoanRecoveryException(e, sys)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Get model evaluation configuration."""
        try:
            art = self.config['artifacts']['model_evaluation']
            thr = self.params['threshold']

            return ModelEvaluationConfig(
                root_dir         = art['root_dir'],
                report_path      = art['report'],
                roc_curve_path   = art['roc_curve'],
                cm_path          = art['confusion_matrix'],
                pr_curve_path    = art['pr_curve'],
                feature_imp_path = art['feature_importance'],
                threshold_high   = thr['high_recovery'],
                threshold_med    = thr['medium_recovery'],
                target_column    = self.schema['target_column'],
            )
        except Exception as e:
            raise LoanRecoveryException(e, sys)

    def get_model_save_config(self) -> ModelSaveConfig:
        """Get model save configuration."""
        try:
            mdl = self.config['models']

            return ModelSaveConfig(
                root_dir           = mdl['root_dir'],
                best_model_path    = mdl['best_model'],
                preprocessor_path  = mdl['preprocessor'],
                threshold_path     = mdl['threshold'],
                feature_names_path = mdl['feature_names'],
                all_models_path    = mdl['all_models'],
            )
        except Exception as e:
            raise LoanRecoveryException(e, sys)
