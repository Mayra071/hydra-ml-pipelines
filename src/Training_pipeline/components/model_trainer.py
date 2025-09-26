import os
import sys
import joblib
import json
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from collections import Counter
import mlflow.sklearn

from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV, ShuffleSplit, StratifiedShuffleSplit, KFold
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
        self.cfg = cfg
        self._model_dir = getattr(getattr(cfg, 'paths', {}), 'models_dir', 'models')
        artifacts_dir = getattr(getattr(cfg, 'paths', {}), 'artifacts_dir', 'artifacts')
        self._metrics_path = os.path.join(artifacts_dir, 'metrics.json')

    def evaluate_model(self, task_type, model, X_test, y_test):
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
            return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def train(self, train_path: str, test_path: str):
        try:
            logger.info("Starting model training with GridSearchCV")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            target_col = self.cfg.dataset.task.target_column

            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            # Select model based on Hydra config
            task_type = self.cfg.dataset.task.type
            model_type = self.cfg.model.model.type
            logger.info(f"Train Data: {X_train.head(2)}")
            logger.info(f"Target Data: {y_train.head(2)}")
           
            if task_type == "classification":
                if model_type == "random_forest":
                    base_model = RandomForestClassifier(random_state=42)
                    logger.info("Random Forest Classifier selected")
                elif model_type == "logistic_regression":
                    base_model = LogisticRegression(max_iter=2000)
                    logger.info("Logistic Regression selected")
                elif model_type == "gradient_boosting":
                    base_model = GradientBoostingClassifier()
                    logger.info("Gradient Boosting Classifier selected")
                else:
                    raise ValueError(f"Unsupported classification model type: {model_type}")
                scoring = self.cfg.training.search.scoring_classification

            elif task_type == "regression":
                if model_type == "random_forest":
                    base_model = RandomForestRegressor(random_state=42)
                    logger.info("Random Forest Regressor selected")
                elif model_type == "gradient_boosting":
                    base_model = GradientBoostingRegressor()
                    logger.info("Gradient Boosting Regressor selected")
                else:
                    raise ValueError(f"Unsupported regression model type: {model_type}")
                scoring = self.cfg.training.search.scoring_regression

            # Prepare GridSearchCV
            model_config_path = f"configs/model/{model_type}.yaml"
            model_cfg = OmegaConf.load(model_config_path)
            param_grid = OmegaConf.to_container(model_cfg.model.params, resolve=True)
            # Validate params
            valid_params = set(base_model.get_params().keys())
            invalid_params = set()
            for param_dict in param_grid:
                if isinstance(param_dict, dict):
                    for key in param_dict.keys():
                        if key not in valid_params:
                            invalid_params.add(key)
            if invalid_params:
                logger.warning(f"Invalid parameters for {model_type}: {invalid_params}. Filtering them out.")
                new_param_grid = []
                for param_dict in param_grid:
                    filtered = {k: v for k, v in param_dict.items() if k in valid_params}
                    if filtered:
                        new_param_grid.append(filtered)
                param_grid = new_param_grid
            
            logger.info(f"Loaded params from {model_type}.yaml: {param_grid}")

            if task_type == "classification":
                counts = Counter(y_train)
                if any(count < 2 for count in counts.values()):
                    cv_strategy = ShuffleSplit(n_splits=self.cfg.training.search.cv, test_size=0.2, random_state=42)
                    logger.info("Using ShuffleSplit due to classes with <2 samples")
                else:
                    cv_strategy = StratifiedShuffleSplit(n_splits=self.cfg.training.search.cv, test_size=0.2, random_state=42)
            else:
                cv_strategy = KFold(n_splits=self.cfg.training.search.cv, shuffle=True, random_state=42)


            # Wire runtime controls
            n_jobs = getattr(self.cfg.training.runtime, 'n_jobs', None)
            verbose = getattr(self.cfg.training.runtime, 'verbose', 0)

            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_strategy,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
            )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            logger.info(f"Best parameters: {search.best_params_}")

            # Evaluate
            metrics = self.evaluate_model(task_type, best_model, X_test, y_test)
            logger.info(f"Final test metrics: {metrics}")

            # Feature importance (for tree-based models)
            if hasattr(best_model, "feature_importances_"):
                importances = best_model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": importances
                }).sort_values(by="importance", ascending=False)
                logger.info(f"Top features:\n{feature_importance_df.head(10)}")

            # Save model to disk for reproducibility
            os.makedirs(self._model_dir, exist_ok=True)
            model_path = os.path.join(self._model_dir, f"{model_type}_{task_type}_best.pkl")
            joblib.dump(best_model, model_path)
            logger.info(f"Best model saved at {model_path}")

            # Save metrics
            os.makedirs(os.path.dirname(self._metrics_path), exist_ok=True)
            results = {"best_params": search.best_params_, "metrics": metrics, "model_path": model_path}
            with open(self._metrics_path, "w") as f:
                json.dump(results, f, indent=4)

            # Log model object to MLflow properly
            mlflow.sklearn.log_model(best_model, artifact_path=f"{model_type}")

            return results

        except Exception as e:
            logger.error("Error during model training", exc_info=True)
            raise CustomException(e, sys)
