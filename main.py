import os
import sys
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from src.logger import logger
from src.exceptions import CustomException

from src.Training_pipeline.components.data_ingestion import DataIngestion
from src.Training_pipeline.components.data_transfer import DataTransformation
from src.Training_pipeline.pipeline.data_preprocessing import Preprocess
from src.Training_pipeline.pipeline.model_training import ModelTrainingPipeline


# Point Hydra to the project-root configs directory
@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    try:
        logger.info("Starting training run")
        logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

        # MLflow setup
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        # Use a simple run name to avoid struct access issues
        run_name = "training_run"
        with mlflow.start_run(run_name=run_name):
            # Log config to MLflow
            mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")

            # 1) Ingest raw data
            ingestion = DataIngestion(cfg)
            raw_path = ingestion.initiate_data_ingestion()

            # 2) Train/test split
            transformer = DataTransformation(cfg)
            train_path, test_path = transformer.split_and_save_data(raw_path)

            # 3) Preprocess (fit on train, transform both)
            preproc = Preprocess(cfg)
            processed_train, processed_test, preproc_obj = preproc.fit_transform_data(train_path, test_path)

            # 4) Train model(s)
            pipeline = ModelTrainingPipeline(cfg, processed_train, processed_test)
            results = pipeline.run()

            # 5) Log results
            if isinstance(results, dict):
                for key, value in results.get("metrics", {}).items():
                    mlflow.log_metric(key, float(value))
                mlflow.log_params(results.get("best_params", {}))
                if "model_path" in results:
                    mlflow.log_artifact(results["model_path"], artifact_path="models")
            mlflow.log_artifact(preproc_obj, artifact_path="artifacts")

            logger.info("Training finished")

    except Exception as e:
        logger.error("Training failed", exc_info=True)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()


