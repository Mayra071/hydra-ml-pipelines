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

        # Dynamically select task config based on task_type (guarded)
        task_key = cfg.task_type
        if not hasattr(cfg.dataset, 'tasks') or task_key not in cfg.dataset.tasks:
            raise KeyError(f"Task '{task_key}' not found in dataset.tasks. Available: {list(getattr(cfg.dataset, 'tasks', {}).keys())}")
        cfg.dataset.task = cfg.dataset.tasks[task_key]

        # MLflow setup (make URI absolute relative to project root)
        from hydra.utils import get_original_cwd
        project_root = get_original_cwd()
        tracking_uri = cfg.mlflow.tracking_uri
        
        mlflow.set_tracking_uri(tracking_uri)
        exp=mlflow.set_experiment(cfg.mlflow.experiment_name)
        logger.info(f"{exp}")

        # Use dynamic run name with model type, task type and timestamp
        import datetime
        run_name = f"{cfg.model.model.type}_{cfg.dataset.task.type}"
        with mlflow.start_run(run_name=run_name):
            # Log config
            mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")

            # Train pipeline
            ingestion = DataIngestion(cfg)
            raw_path = ingestion.initiate_data_ingestion()
            transformer = DataTransformation(cfg)
            train_path, test_path = transformer.split_and_save_data(raw_path)
            preproc = Preprocess(cfg)
            processed_train, processed_test, preproc_path = preproc.fit_transform_data(train_path, test_path)

            pipeline = ModelTrainingPipeline(cfg, processed_train, processed_test)
            results = pipeline.run()

            # Log results
            if isinstance(results, dict):
                mlflow.log_metrics(results.get("metrics", {}))
                mlflow.log_params(results.get("best_params", {}))

                if "model_path" in results:
                    mlflow.log_artifact(results["model_path"], artifact_path="models")

            # Log preprocessor artifact produced by preprocessing step
            os.makedirs("artifacts", exist_ok=True)
            mlflow.log_artifact(preproc_path, artifact_path="artifacts")

            # Log dataset file
            mlflow.log_artifact(cfg.dataset.paths.dataset, artifact_path="datasets")
            logger.info("Training finished")



    except Exception as e:
        logger.error("Training failed", exc_info=True)
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()


