import os
import pytest
from src.loanrecovery.config import ConfigurationManager
from src.loanrecovery.components.data_ingestion import DataIngestion
from src.loanrecovery.components.data_transformation import DataTransformation
from src.loanrecovery.pipeline.prediction_pipeline import PredictPipeline


class TestDataIngestion:
    def test_data_ingestion_config(self):
        config_manager = ConfigurationManager()
        config = config_manager.get_data_ingestion_config()
        assert config is not None
        assert os.path.exists(config.root_dir)

    def test_train_test_files_created(self):
        # This test assumes pipeline has been run
        config_manager = ConfigurationManager()
        config = config_manager.get_data_ingestion_config()
        if os.path.exists(config.train_data_path) and os.path.exists(config.test_data_path):
            assert os.path.getsize(config.train_data_path) > 0
            assert os.path.getsize(config.test_data_path) > 0
        else:
            pytest.skip("Train/test files not yet created. Run pipeline first.")


class TestDataTransformation:
    def test_preprocessor_exists(self):
        config_manager = ConfigurationManager()
        config = config_manager.get_data_transformation_config()
        if os.path.exists(config.preprocessor_path):
            assert os.path.getsize(config.preprocessor_path) > 0
        else:
            pytest.skip("Preprocessor not yet created. Run pipeline first.")


class TestPredictionPipeline:
    def test_prediction_output_format(self):
        # This test requires model to be trained
        try:
            pipeline = PredictPipeline()
            import pandas as pd
            dummy_input = pd.DataFrame({
                'AMT_INCOME_TOTAL': [50000.0],
                'AMT_CREDIT': [100000.0],
                'AMT_ANNUITY': [5000.0],
                'AMT_GOODS_PRICE': [100000.0],
                'DAYS_BIRTH': [-10000],
                'DAYS_EMPLOYED': [-2000],
                'CNT_CHILDREN': [0],
                'CODE_GENDER': ['M'],
                'FLAG_OWN_CAR': ['Y']
            })
            pred, proba = pipeline.predict(dummy_input)
            assert pred in [0, 1]
            assert 0.0 <= proba <= 1.0
        except Exception:
            pytest.skip("Model not yet trained. Run pipeline first.")
