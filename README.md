# Writing the polished README content to a markdown file

# 🏗️ ML Training Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)  
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org/stable/)  
[![MLflow](https://img.shields.io/badge/MLflow-2.0%2B-red)](https://mlflow.org/)  
[![Hydra](https://img.shields.io/badge/Hydra-1.3%2B-green)](https://hydra.cc/)

An **end-to-end machine learning training pipeline** for the California Housing dataset featuring configurable workflows, experiment tracking, and modular components.

## 📋 Table of Contents
- [🚀 Features](#-features)  
- [📁 Project Structure](#-project-structure)  
- [🛠️ Workflow Diagram](#-workflow-diagram)  
- [💻 Installation](#-installation)  
- [🏃 Quick Start](#-quick-start)  
- [⚙️ Configuration](#-configuration)  
- [📚 Usage Examples](#-usage-examples)  
- [📊 Outputs](#-outputs)  
- [🔬 MLflow Tracking](#-mlflow-tracking)  
- [🐛 Troubleshooting](#-troubleshooting)  
- [🤝 Contributing](#-contributing)  
- [📄 License](#-license)  
- [🙏 Acknowledgments](#-acknowledgments)  

## 🚀 Features
- 🔧 **Configurable Tasks**: Regression (predicting median house value) and classification (price categories).  
- 🧩 **Modular Pipeline**: Separate components for data ingestion, preprocessing, model training, and evaluation.  
- 🤖 **Multiple Models**: Random Forest, Logistic Regression, Gradient Boosting with hyperparameter tuning.  
- 📊 **Experiment Tracking**: MLflow integration for metrics, parameters, and artifacts.  
- ⚡ **Automated Preprocessing**: Smart handling of missing values, scaling, and encoding based on configuration.  
- 📈 **Performance Monitoring**: Cross-validation with multiple scoring metrics.  

## 💻 Installation
### Prerequisites
- Python 3.8+  
- pip package manager  

### Step-by-Step Setup
\`\`\`bash
# Clone the repository
git clone <repository-url>
cd ml-training-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### Prepare the Dataset
- Download the California Housing dataset.  
- Place it at: C:\\Company\\Data_Set\\housing.csv *(or update path in config)*  
- Optional: use built-in dataset loader in src/utils.py.

## 🏃 Quick Start
### Regression (Default: Random Forest)
\`\`\`bash
python main.py
\`\`\`

### Classification (Logistic Regression)
\`\`\`bash
python main.py dataset.task_type=classification model=logistic_regression
\`\`\`

### View Results in MLflow
\`\`\`bash
mlflow ui --backend-store-uri mlruns/
\`\`\`
Open [http://localhost:5000](http://localhost:5000) in browser.

## ⚙️ Configuration
### Dataset Config
\`\`\`yaml
dataset:
  name: "california_housing"
  task_type: "regression"  # or "classification"
  target_column: "median_house_value"
  test_size: 0.2
  random_state: 42
\`\`\`

### Model Options
- random_forest → Tree-based ensemble  
- logistic_regression → Linear classifier  
- gradient_boosting → Boosted ensemble  

### Preprocessing Config
\`\`\`yaml
preprocessing:
  numeric:
    imputer: "median"
    scaler: "standard"   # standard, minmax, robust
  categorical:
    encoder: "onehot"
\`\`\`

## 📚 Usage Examples
### Regression
\`\`\`bash
python main.py model.model.type=gradient_boosting
python main.py model.random_forest.hyperparameters.n_estimators=500
\`\`\`

### Classification
\`\`\`bash
python main.py dataset.task_type=classification dataset.classification.n_quantiles=3
python main.py training.scoring.classification='["accuracy","f1_macro"]'
\`\`\`

### Advanced
\`\`\`bash
python main.py preprocessing.numeric.scaler=minmax preprocessing.categorical.encoder=ordinal
python main.py dataset.test_size=0.3 training.cv_folds=3
\`\`\`

## 📊 Outputs
- artifacts/ → processed datasets, preprocessing objects, metrics  
- models/ → trained models (.pkl)  
- logs/ → detailed training logs  
- mlruns/ → MLflow experiment tracking  

## 🔬 MLflow Tracking
\`\`\`bash
mlflow ui --backend-store-uri mlruns/
mlflow ui --backend-store-uri mlruns/ --experiment-name "housing_prediction"
\`\`\`

## 🐛 Troubleshooting
```bash
# Dataset not found
python main.py paths.dataset_path="/new/path/housing.csv"

# Memory issues
python main.py dataset.test_size=0.1 model.random_forest.hyperparameters.max_depth=10

# MLflow errors
rm -rf mlruns/
python main.py

# Debug mode
python main.py logging.level=DEBUG


