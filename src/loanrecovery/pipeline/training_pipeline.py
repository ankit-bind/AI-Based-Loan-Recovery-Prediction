import time
import gc
import os
from src.loanrecovery.config import ConfigurationManager
from src.loanrecovery.components.data_ingestion import DataIngestion
from src.loanrecovery.components.data_validation import DataValidation
from src.loanrecovery.components.feature_engineering import FeatureEngineering
from src.loanrecovery.components.data_transformation import DataTransformation
from src.loanrecovery.components.model_trainer import ModelTrainer
from src.loanrecovery.components.model_evaluation import ModelEvaluation
from src.loanrecovery.components.explainability import Explainability
from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException
import sys

logger = get_logger(__name__)

class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run_pipeline(self):
        try:
            logger.info("="*60)
            logger.info('Loan Recovery Training Pipeline Started')
            logger.info("="*60)
            
            pipeline_results = {}

            # Stage 1: Data Ingestion
            logger.info("STAGE 1: Data Ingestion")
            start_time = time.time()
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            train_path, test_path = data_ingestion.initiate_data_ingestion()
            logger.info(f"Raw train data: {train_path}")
            logger.info(f"Raw test data: {test_path}")
            logger.info(f'Stage completed in {time.time() - start_time:.2f} seconds')
            gc.collect()

            # Stage 2: Data Validation
            logger.info("STAGE 2: Data Validation")
            start_time = time.time()
            data_validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            validation_status = data_validation.initiate_data_validation(data_path=train_path)
            logger.info(f"Validation status: {validation_status}")
            
            # Validation stop condition
            if not validation_status:
                raise ValueError('Data validation failed. Pipeline stopped.')
            
            logger.info(f'Stage completed in {time.time() - start_time:.2f} seconds')
            gc.collect()

            # Stage 3: Feature Engineering
            logger.info("STAGE 3: Feature Engineering")
            start_time = time.time()
            fe_config = self.config_manager.get_data_ingestion_config()  # Reuse for paths
            feature_engineering = FeatureEngineering(config=fe_config)
            raw_app_path = train_path
            raw_bureau_path = data_ingestion_config.bureau
            
            engineered_path = feature_engineering.initiate_feature_engineering(
                application_path=raw_app_path, bureau_path=raw_bureau_path
            )
            logger.info(f"Engineered dataset: {engineered_path}")
            logger.info(f'Stage completed in {time.time() - start_time:.2f} seconds')
            gc.collect()

            # Stage 4: Data Transformation
            logger.info("STAGE 4: Data Transformation")
            start_time = time.time()
            data_transformation_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            transformed_path = data_transformation.initiate_data_transformation(
                engineered_data_path=engineered_path
            )
            logger.info(f"Transformed data: {transformed_path}")
            logger.info(f'Stage completed in {time.time() - start_time:.2f} seconds')
            gc.collect()

            # Stage 5: Model Training
            logger.info("STAGE 5: Model Training")
            start_time = time.time()
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_path, test_auc = model_trainer.initiate_model_trainer(
                engineered_data_path=transformed_path
            )
            logger.info(f"Best model saved to: {model_path}, Test AUC: {test_auc:.4f}")
            logger.info(f'Stage completed in {time.time() - start_time:.2f} seconds')
            gc.collect()

            # Stage 6: Model Evaluation
            logger.info("STAGE 6: Model Evaluation")
            start_time = time.time()
            model_evaluation_config = self.config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            metrics = model_evaluation.initiate_model_evaluation(
                engineered_data_path=transformed_path
            )
            logger.info(f"Final evaluation metrics: {metrics}")
            logger.info(f'Stage completed in {time.time() - start_time:.2f} seconds')
            gc.collect()

            # Stage 7: Explainability
            logger.info("STAGE 7: Explainability")
            start_time = time.time()
            explainability = Explainability(config=model_evaluation_config)
            try:
                shap_path = explainability.initiate_explainability(
                    engineered_data_path=transformed_path, 
                    output_dir=model_evaluation_config.root_dir
                )
                logger.info(f"SHAP plot saved to: {shap_path}")
            except Exception as shap_error:
                logger.warning(f'SHAP failed: {shap_error}')
                shap_path = None
            logger.info(f'Stage completed in {time.time() - start_time:.2f} seconds')
            gc.collect()

            # Pipeline artifact validation
            logger.info("Validating critical artifacts...")
            required_files = [
                'models/best_model.pkl',
                'models/threshold.pkl',
                'models/feature_names.json'
            ]
            for file in required_files:
                if not os.path.exists(file):
                    raise FileNotFoundError(f'Missing artifact: {file}')
            logger.info('All critical artifacts validated successfully')

            # Final Pipeline Summary
            logger.info("="*60)
            logger.info('FINAL PIPELINE SUMMARY')
            logger.info(f'Best Model ROC-AUC: {test_auc:.4f}')
            logger.info(f'SHAP Output: {shap_path}')
            logger.info("="*60)
            logger.info('Loan Recovery Training Pipeline Completed Successfully')
            logger.info("="*60)
            
            # Return artifact registry
            return {
                'model_path': model_path,
                'roc_auc': test_auc,
                'shap_path': shap_path,
                'metrics': metrics,
                'engineered_path': engineered_path,
                'transformed_path': transformed_path,
                'train_path': train_path,
                'test_path': test_path
            }

        except Exception as e:
            raise LoanRecoveryException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
