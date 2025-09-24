import os
import logging
from datetime import datetime

# Absolute path to logs folder
project_root = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(project_root, "..", "logs")
os.makedirs(logs_dir, exist_ok=True)

LOG_FILE = f"Training_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Disable root logger to prevent console output
logging.getLogger().handlers.clear()
logging.getLogger().setLevel(logging.WARNING)  # Set to WARNING to suppress INFO and DEBUG

# Create dedicated logger
logger = logging.getLogger("TrainingPipelineLogger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger

# Remove any existing handlers from the custom logger
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# File handler only
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add handler (file only, no console)
logger.addHandler(file_handler)

