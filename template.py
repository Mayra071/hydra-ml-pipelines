import os
from pathlib import Path
from src.logger import logger

project_name = "Training_pipeline"

file_paths=[
    # f"src/{project_name}",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transfer.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/data_preprocessing.py",
    f"src/{project_name}/pipeline/model_training.py",
    f"src/{project_name}/pipeline/model_evaluation.py",
    f"src/{project_name}/config.py",
    f"src/{project_name}/main.py",
    f"configs/config.yaml",
    f"src/exceptions.py",
    f"src/utils.py",
    f"src/logger.py",
    "app.py",
    "requirements.txt",
    "README.md",
    "setup.py",
    # "Dockerfile"
]

for filepath in file_paths:
    filepath=Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logger.info(f"Created directory: {filedir}")

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
        logger.info(f"Created file: {filepath}")
    else:
        logger.info(f"File already exists: {filepath}")
        
    