# src/config/settings.py
"""
Configuration settings for the materials ML pipeline
"""
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    FEATURES_DATA_DIR: Path = DATA_DIR / "features"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"   
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = Field("http://localhost:5000", env="MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME: str = Field("materials_bandgap_prediction", env="MLFLOW_EXPERIMENT_NAME")
    
    # Prefect Configuration
    PREFECT_API_URL: str = Field("http://localhost:4200/api", env="PREFECT_API_URL")
    
    # Feast Configuration
    FEAST_REPO_PATH: Path = PROJECT_ROOT / "feast_repo"
    
    # Data Processing    
    TEST_SIZE: float = Field(0.2, description="Test set size")
    RANDOM_STATE: int = Field(42, description="Random state for reproducibility")
    
    # Model Configuration
    CV_FOLDS: int = Field(5, description="Number of cross-validation folds")
    N_TRIALS: int = Field(100, description="Number of hyperparameter optimization trials")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

# Create directories if they don't exist
for directory in [
    settings.DATA_DIR,
    settings.RAW_DATA_DIR,
    settings.PROCESSED_DATA_DIR,
    settings.FEATURES_DATA_DIR,
    settings.MODELS_DIR,
    settings.LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
# Mlflow command
# mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
