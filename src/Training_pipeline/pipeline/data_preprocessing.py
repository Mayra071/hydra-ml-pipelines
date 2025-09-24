import os
import sys
import joblib
import pandas as pd

from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.logger import logger
from src.exceptions import CustomException


@dataclass
class DataPreprocessingConfig:
    processed_train_path: str = os.path.join("artifacts", "train_processed.csv")
    processed_test_path: str = os.path.join("artifacts", "test_processed.csv")
    processed_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")
    label_encoder_path: str = os.path.join("artifacts", "label_encoder.pkl")


class Preprocess:
    def __init__(self, cfg):
        self.cfg = cfg
        self.preprocess_cfg = DataPreprocessingConfig()

    def create_data_preprocessing(self, feature_df: pd.DataFrame):
        try:
            numerical_cols = feature_df.select_dtypes(include=['int64', 'float64']).columns.to_list()
            categorical_cols = feature_df.select_dtypes(include=['object']).columns.to_list()

            logger.info(f'Numerical columns: {numerical_cols}')
            logger.info(f'Categorical columns: {categorical_cols}')

            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            logger.error("Error in creating preprocessing pipeline")
            raise CustomException(e, sys)

    def fit_transform_data(self, train_path, test_path):
        logger.info("Starting data preprocessing")
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            task_type = self.cfg.dataset.task.type
            target_col = self.cfg.dataset.task.target_column
            original_target = None

            # Handle classification labeling if specified
            if task_type == 'classification' and 'labeling' in self.cfg.dataset.task:
                src_col = self.cfg.dataset.task.labeling.source_column
                method = self.cfg.dataset.task.labeling.method
                custom_labels = getattr(self.cfg.dataset.task.labeling, 'labels', None)

                if method == 'quantile':
                    q = getattr(self.cfg.dataset.task.labeling, 'q')

                    if isinstance(q, (list, tuple)):  # Multi-class
                        thresholds = train_df[src_col].quantile(q).tolist()
                        bins = [-float("inf")] + thresholds + [float("inf")]

                        if custom_labels:
                            if len(custom_labels) != len(bins) - 1:
                                raise ValueError(
                                    f"Number of labels ({len(custom_labels)}) must match number of bins ({len(bins)-1})")
                            labels = custom_labels
                        else:
                            labels = [f"class_{i}" for i in range(len(bins) - 1)]

                        train_df[target_col] = pd.cut(train_df[src_col], bins=bins, labels=labels)
                        test_df[target_col] = pd.cut(test_df[src_col], bins=bins, labels=labels)

                        # Encode with LabelEncoder
                        le = LabelEncoder()
                        train_df[target_col] = le.fit_transform(train_df[target_col])
                        test_df[target_col] = le.transform(test_df[target_col])

                        # Save LabelEncoder
                        joblib.dump(le, self.preprocess_cfg.label_encoder_path)
                        logger.info(f"LabelEncoder saved at {self.preprocess_cfg.label_encoder_path}")

                    else:  # Binary
                        threshold = train_df[src_col].quantile(float(q))
                        train_df[target_col] = (train_df[src_col] >= threshold).astype(int)
                        test_df[target_col] = (test_df[src_col] >= threshold).astype(int)

                    original_target = src_col

                elif method == 'threshold':
                    threshold = float(getattr(self.cfg.dataset.task.labeling, 'threshold'))
                    train_df[target_col] = (train_df[src_col] >= threshold).astype(int)
                    test_df[target_col] = (test_df[src_col] >= threshold).astype(int)
                    original_target = src_col

                else:
                    raise ValueError(f"Unsupported labeling method: {method}")

            # Prepare feature matrices
            drop_cols = [target_col]
            if original_target and original_target != target_col:
                drop_cols.append(original_target)

            X_train = train_df.drop(columns=drop_cols, errors='ignore')
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=drop_cols, errors='ignore')
            y_test = test_df[target_col]

            # Preprocessing
            preprocessor = self.create_data_preprocessing(X_train)
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Build column names
            feature_names = []
            for name, trans, cols in preprocessor.transformers_:
                if name == 'num':
                    feature_names.extend(cols)
                else:
                    # one-hot encoder column names
                    cat_cols = trans.named_steps['encoder'].get_feature_names_out(cols)
                    feature_names.extend(cat_cols)

            # Convert back to DataFrame
            train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
            test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

            # Append target
            train_processed_df[target_col] = y_train.values
            test_processed_df[target_col] = y_test.values

            # Save datasets
            os.makedirs(os.path.dirname(self.preprocess_cfg.processed_train_path), exist_ok=True)
            train_processed_df.to_csv(self.preprocess_cfg.processed_train_path, index=False)
            test_processed_df.to_csv(self.preprocess_cfg.processed_test_path, index=False)

            logger.info(f"Processed train saved at {self.preprocess_cfg.processed_train_path}")
            logger.info(f"Processed test saved at {self.preprocess_cfg.processed_test_path}")
            logger.info(f'Train head: \n{train_processed_df.head(2)}')
            logger.info(f'Test head: \n{test_processed_df.head(2)}')

            # Save preprocessor
            joblib.dump(preprocessor, self.preprocess_cfg.processed_obj_path)
            logger.info(f"Preprocessor saved at {self.preprocess_cfg.processed_obj_path}")

            return (
                self.preprocess_cfg.processed_train_path,
                self.preprocess_cfg.processed_test_path,
                self.preprocess_cfg.processed_obj_path
            )

        except Exception as e:
            logger.error("Exception during data preprocessing")
            raise CustomException(e, sys)
