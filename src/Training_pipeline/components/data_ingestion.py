import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logger
from src.utils import read_data
from src.exceptions import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ingestion_conf = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Initiating Data Ingestion")
        try:
            # Read dataset
            logger.info("Reading the dataset")
            df = read_data(self.cfg)

            # Ensure artifacts folder exists
            os.makedirs("artifacts", exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_conf.raw_data_path, index=False, header=True)
            logger.info(f"Raw data saved at {self.ingestion_conf.raw_data_path}")

            return self.ingestion_conf.raw_data_path

        except Exception as e:
            logger.error("Exception during data ingestion", exc_info=True)
            raise CustomException(e, sys)
