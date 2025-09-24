import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logger
from src.exceptions import CustomException

@dataclass
class DataTransformationConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataTransformation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.transformation_conf = DataTransformationConfig()

    def split_and_save_data(self,path):
        logger.info("Starting data transformation (train-test split)")
        try:
            # Load raw data
            df = pd.read_csv(path)
            logger.info(f'Display some data :\n {df.head(2)}')
            # Train-test split (from training config group)
            test_size = self.cfg.training.split.test_size
            random_state = self.cfg.training.split.random_state
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

            # Save split data
            train_df.to_csv(self.transformation_conf.train_data_path, index=False, header=True)
            test_df.to_csv(self.transformation_conf.test_data_path, index=False, header=True)

            logger.info(f"Train data saved at {self.transformation_conf.train_data_path}")
            logger.info(f"Test data saved at {self.transformation_conf.test_data_path}")

            return self.transformation_conf.train_data_path, self.transformation_conf.test_data_path

        except Exception as e:
            logger.error("Exception during data transformation", exc_info=True)
            raise CustomException(e, sys)
