ML Training Pipeline
https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/Scikit--learn-1.0%252B-orange
https://img.shields.io/badge/MLflow-2.0%252B-red
https://img.shields.io/badge/Hydra-1.3%252B-green

An end-to-end machine learning training pipeline for the California Housing dataset featuring configurable workflows, experiment tracking, and modular components.


🚀 Features
🔧 Configurable Tasks: Support for both regression (predicting median house value) and classification (price categories)

🧩 Modular Pipeline: Separate components for data ingestion, preprocessing, model training, and evaluation

🤖 Multiple Models: Random Forest, Logistic Regression, Gradient Boosting with hyperparameter tuning

📊 Experiment Tracking: Comprehensive MLflow integration for metrics, parameters, and artifacts

⚡ Automated Preprocessing: Smart handling of missing values, scaling, and encoding based on configuration

📈 Performance Monitoring: Cross-validation with multiple scoring metrics

📁 Project Structure
text
ml-training-pipeline/
├── 📂 configs/                 # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── 📂 dataset/            # Dataset-specific configs
│   ├── 📂 model/              # Model hyperparameters
│   ├── 📂 preprocessing/      # Preprocessing strategies
│   └── 📂 training/           # Training parameters
├── 📂 src/Training_pipeline/  # Core pipeline code
│   ├── 📂 components/         # Data ingestion, transformation, model training
│   └── 📂 pipeline/           # Pipeline orchestration
├── 📂 artifacts/              # Processed data and preprocessing objects
├── 📂 models/                 # Saved trained models
├── 📂 mlruns/                 # MLflow experiment tracking
├── 📂 logs/                   # Training logs
└── 📂 tests/                  # Test cases
💻 Installation
Prerequisites
Python 3.8 or higher

pip package manager

Step-by-Step Setup
Clone the repository

bash
git clone <repository-url>
cd ml-training-pipeline
Create virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Prepare the dataset

Download the California Housing dataset

Place it at C:\Company\Data_Set\housing.csv (or update path in config)

Alternatively, use the built-in dataset loader

🏃 Quick Start
Basic Regression Task
bash
# Run with default settings (Random Forest regression)
python main.py
Classification Task
bash
# Run classification with Logistic Regression
python main.py dataset.task_type=classification model=logistic_regression
View Results
bash
# Start MLflow UI to track experiments
mlflow ui --backend-store-uri mlruns/
Then open http://localhost:5000 in your browser.

⚙️ Configuration
Dataset Configuration
yaml
dataset:
  name: "california_housing"
  task_type: "regression"  # or "classification"
  target_column: "median_house_value"
  test_size: 0.2
  random_state: 42
Model Configuration
Choose from:

random_forest: Ensemble method with tree-based learning

logistic_regression: Linear model for classification

gradient_boosting: Sequential ensemble with boosting

# Gradient Boosting for regression
python main.py model.model.type=gradient_boosting

# Custom hyperparameters
python main.py 

# Classification / Regression 
python main.py dataset.task_type=Y model.model.type= X

📊 Outputs
Generated Artifacts
artifacts/: Processed datasets, preprocessing objects, metrics

models/: Trained model with best hyperparameters

logs/: Detailed training logs

mlruns/: MLflow experiment tracking data

🔬 MLflow Tracking
Accessing Experiments
bash
# Start MLflow UI
mlflow ui --backend-store-uri mlruns/

# View specific experiment
mlflow ui --backend-store-uri mlruns/ --experiment-name "housing_prediction"
Key Tracking Features
Parameter logging and hyperparameter tracking

Multiple evaluation metrics across folds

Model files and preprocessing objects storage

Experiment comparison capabilities

🐛 Troubleshooting
Common Issues
Dataset Not Found

bash
python main.py paths.dataset_path="/new/path/housing.csv"
Memory Issues

bash
python main.py dataset.test_size=0.1 model.random_forest.hyperparameters.max_depth=10
MLflow Tracking Errors

bash
rm -rf mlruns/
python main.py
Debug Mode

bash
python main.py logging.level=DEBUG
🤝 Contributing
Adding New Models
Create config in configs/model/new_model.yaml

Update ModelTrainer component

Add tests in tests/test_pipeline.py

Running Tests
bash
pytest tests/
pytest tests/test_pipeline.py::test_data_ingestion -v
📄 License
This project is licensed under the MIT License.

🙏 Acknowledgments
Name: Manish Kumar
California Housing Dataset: UCI Machine Learning Repository

Hydra: Flexible configuration management

MLflow: Machine learning lifecycle management

Scikit-learn: Machine learning algorithms and utilities

Need Help? Check the troubleshooting section or examine generated log files.

Found a Bug? Please open an issue with configuration used and error details.

