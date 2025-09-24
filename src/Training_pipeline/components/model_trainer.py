import os
import sys
import joblib
import json
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from src.logger import logger
from src.exceptions import CustomException

@dataclass
class ModelTrainerConfig:
    model_dir: str = "models"
    metrics_path: str = os.path.join("artifacts", "metrics.json")


class ModelTrainer:
    def __init__(self, cfg):
        # Keep original cfg but compute output paths with safe defaults
        self.cfg = cfg  
        self._model_dir = getattr(getattr(cfg, 'paths', {}), 'models_dir', 'models')
        artifacts_dir = getattr(getattr(cfg, 'paths', {}), 'artifacts_dir', 'artifacts')
        self._metrics_path = os.path.join(artifacts_dir, 'metrics.json')

    def evaluate_model(self, task_type, model, X_test, y_test):
        """Evaluate model by task type and return metrics."""
        y_pred = model.predict(X_test)
        if task_type == "classification":
            return {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
        elif task_type == "regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        else:
            raise ValueError(f"Unsupported task type for evaluation: {task_type}")

    def train(self, train_path: str, test_path: str):
        try:
            logger.info("Starting model training with GridSearchCV")

            # Load processed data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = self.cfg.dataset.task.target_column
            X_train = train_df.drop(columns=target_col)
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=target_col)
            y_test = test_df[target_col]

            # Select model based on Hydra config
            task_type = self.cfg.dataset.task.type
            model_type = self.cfg.model.model.type

            if task_type == "classification":
                if model_type == "random_forest":
                    base_model = RandomForestClassifier(random_state=42)
                    logger.info("Random Forest \n")
                elif model_type == "logistic_regression":
                    base_model = LogisticRegression(max_iter=1000)
                    logger.info("Logistic Regression \n")
                elif model_type == "gradient_boosting":
                    base_model = GradientBoostingClassifier()
                    logger.info("Gradiant Boosting \n")
                else:
                    raise ValueError(f"Unsupported classification model type: {model_type}")
                scoring = self.cfg.training.search.scoring_classification
            elif task_type == "regression":
                if model_type == "random_forest":
                    base_model = RandomForestRegressor(random_state=42)
                    logger.info("Random Forest \n")
                elif model_type == "gradient_boosting":
                    base_model = GradientBoostingRegressor()
                    logger.info("Gradiant Boosting \n")
                else:
                    raise ValueError(f"Unsupported regression model type: {model_type}")
                scoring = self.cfg.training.search.scoring_regression
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            # Convert DictConfig to Python dict
            from omegaconf import OmegaConf
            valid_params = base_model.get_params().keys()           
            cfg_params = OmegaConf.to_container(self.cfg.model.model.params, resolve=True)

            param_grid = {k: v for k, v in cfg_params.items() if k in valid_params}

            # GridSearchCV
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=self.cfg.training.search.cv,
                scoring=scoring,
                n_jobs=self.cfg.training.runtime.n_jobs,
                verbose=self.cfg.training.runtime.verbose,
            )


            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            logger.info(f"Best params: {search.best_params_}")

            # Evaluate on test set
            metrics = self.evaluate_model(task_type, best_model, X_test, y_test)
            logger.info(f"Final test metrics: {metrics}")

            # Save model
            os.makedirs(self._model_dir, exist_ok=True)
            model_path = os.path.join(self._model_dir, f"{self.cfg.model.model.type}_best.pkl")
            joblib.dump(best_model, model_path)
            logger.info(f"Best model saved at {model_path}")

            # Save metrics
            results = {
                "best_params": search.best_params_,
                "metrics": metrics,
                "model_path": model_path
            }

            os.makedirs(os.path.dirname(self._metrics_path), exist_ok=True)
            with open(self._metrics_path, "w") as f:
                json.dump(results, f, indent=4)

            return results

        except Exception as e:
            logger.error("Error during model training with RandomizedSearchCV", exc_info=True)
            raise CustomException(e, sys)
