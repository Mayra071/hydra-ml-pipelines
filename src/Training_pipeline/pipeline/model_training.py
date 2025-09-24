from src.Training_pipeline.components.model_trainer import ModelTrainer
from src.logger import logger
from src.exceptions import CustomException
import sys
class ModelTrainingPipeline:
    def __init__(self, cfg, train_path: str, test_path: str):
        self.cfg = cfg
        self.train_path = train_path
        self.test_path = test_path

    def run(self):
        try:
            logger.info("Launching Model Training Pipeline with Hydra config")
            trainer = ModelTrainer(self.cfg)
            results = trainer.train(self.train_path, self.test_path)
            logger.info(f"Training finished. Results: {results}")
            return results
        except Exception as e:
            logger.error("Pipeline failed", exc_info=True)
            raise CustomException(e,sys)
