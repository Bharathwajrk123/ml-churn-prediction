"""
Configuration module for the ML project.
Handles environment variables and project settings.
"""

import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# FastAPI Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", 8000))
API_TITLE = "Customer Churn Prediction API"
API_DESCRIPTION = "An API that predicts customer churn based on Telco data."
API_VERSION = "1.0.0"

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlruns/mlflow.db"
)
MLFLOW_BACKEND_STORE_URI = os.getenv(
    "MLFLOW_BACKEND_STORE_URI",
    "sqlite:///mlruns/mlflow.db"
)
MLFLOW_DEFAULT_ARTIFACT_ROOT = os.getenv(
    "MLFLOW_DEFAULT_ARTIFACT_ROOT",
    "/mlruns"
)
MLFLOW_EXPERIMENT_NAME = "Customer_Churn_Prediction"

# Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
DATA_PATH = os.getenv("DATA_PATH", "data/churn.csv")

# Dataset Configuration
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model Hyperparameters
MODEL_HYPERPARAMETERS = {
    "n_estimators": int(os.getenv("N_ESTIMATORS", 100)),
    "max_depth": int(os.getenv("MAX_DEPTH", 5)),
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def get_config() -> dict:
    """
    Get all configuration as a dictionary.
    
    Returns:
        dict: Configuration dictionary
    """
    return {
        "environment": ENVIRONMENT,
        "debug": DEBUG,
        "api_host": API_HOST,
        "api_port": API_PORT,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
        "model_hyperparameters": MODEL_HYPERPARAMETERS,
    }
