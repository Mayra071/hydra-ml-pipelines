import os
import sys
from src.logger import logger
from src.exceptions import CustomException
import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

import pandas as pd
import numpy as np

# function to read data set

def read_data(cfg: DictConfig):

    try:

        # Support either flattened paths.dataset or nested dataset.paths.dataset
        dataset_path = OmegaConf.select(cfg, "paths.dataset")
        if dataset_path is None:
            dataset_path = OmegaConf.select(cfg, "dataset.paths.dataset")
        if dataset_path is None:
            raise KeyError("Dataset path not found in cfg.paths.dataset or cfg.dataset.paths.dataset")
        # Resolve relative path against original working directory (project root)
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(get_original_cwd(), dataset_path)
        df = pd.read_csv(dataset_path)
        logger.info(f"Data set readed succesfully from {dataset_path}")
        return df
        
        
    except Exception as e:
        logger.info("Handel Exception at read data in utils file")
        raise CustomException(e,sys)
