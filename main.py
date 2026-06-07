from src.loanrecovery.pipeline.training_pipeline import TrainingPipeline
from src.loanrecovery.logger import get_logger
from src.loanrecovery.exception import LoanRecoveryException
import sys

logger = get_logger(__name__)

if __name__ == "__main__":
    try:
        logger.info("Running main.py")
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        logger.info("main.py completed successfully")
    except Exception as e:
        logger.error("Error in main.py")
        raise LoanRecoveryException(e, sys)
