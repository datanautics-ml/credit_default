# flows/predict_flow.py
"""
ML training pipeline for materials bandgap prediction
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_table_artifact, create_markdown_artifact
import mlflow
import json
import os
import sys
# Add src to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(project_root))
from src.config.settings import settings
from src.models.training import  ModelTrainer
from src.utils.logging import setup_logging
import joblib

@task(name="Load Data ", retries=1)
def load_new_data() -> pd.DataFrame:
    """Load lastet data from CSV file"""
    logger = get_run_logger()
    logger.info((f"Checking processed data from {settings.PROCESSED_DATA_DIR}"))
    processed_files = list(settings.PROCESSED_DATA_DIR.glob("data_*.csv"))
    if not processed_files:
        latest_file = 'data_01.csv'
    else:    
        latest_file = sorted(processed_files)[-1].name
        next_idx = int(latest_file.split('_')[-1].split('.')[0]) + 1
        next_idx_str = f"{next_idx:02d}"
        latest_file = f"data_{next_idx_str}.csv"
    latest_file_path = settings.DATA_DIR / latest_file
    logger.info(f"Latest processed data file: {latest_file_path}")
    
    logger.info(f"Loading data from {settings.DATA_DIR}")
    data = pd.read_csv(latest_file_path, header=0)
    logger.info(f"Data shape: {data.shape}")
    # Copy the latest file to the processed data directory
    try:
        settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = settings.PROCESSED_DATA_DIR / latest_file
        if not dest_path.exists():
            dest_path.write_bytes(latest_file_path.read_bytes())
            logger.info(f"Copied {latest_file_path} to {dest_path}")
        else:
            logger.info(f"Processed file already exists at {dest_path}")
    except Exception as e:
        logger.error(f"Failed to copy file to processed dir: {e}")
        raise
    return data

@task(name="Load Model")
def load_model() -> Any:
    """Load  ML model from disk"""
    logger = get_run_logger()
    logger.info(f"Loading model from {settings.MODELS_DIR}")

    with settings.MODELS_DIR.joinpath("training_results.json").open("r") as f:
        training_results = json.load(f)
    
    best_model_name = None
    best_score = float('-inf')    
    for model_name, results in training_results.items():
        score = training_results[model_name]['f1']
        if score > best_score:
            best_score = score
            best_model_name = model_name    
    model_path = settings.MODELS_DIR / f"{best_model_name}_model.joblib"    
    logger.info(f"Best model: {best_model_name}")
    
    try:
        model = joblib.load(model_path)
        if best_model_name in ['logistic_regression', 'svc', 'ridge']:
            scaler_path = settings.MODELS_DIR / "standard_scaler.joblib"
            scaler = joblib.load(scaler_path)            
            logger.info("Model loaded successfully")
            return model, scaler
        else:
            logger.info("Model loaded successfully")
            return model, None            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@task(name="Make Predictions")
def make_predictions(model: Any, dataset: pd.DataFrame, scaler: Any = None) -> pd.DataFrame:
    """Make predictions using the loaded model"""
    logger = get_run_logger()
    logger.info("Making predictions")
    # dataset.drop(columns=["MONTH"], errors='ignore', inplace=True)
    features = [col for col in dataset.columns if col not in ["default payment next month", "MONTH", "ID"]]
    if scaler:
        X = dataset[features]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
    else:
        X = dataset[features]
        predictions = model.predict(X)
    dataset['predictions'] = predictions
    
    logger.info("Predictions completed")
    return dataset

@flow(name="ML Model Training Flow")
def ml_predict_flow(
    
) -> None:
    """
    Prefect flow to orchestrate ML model prediction
    
    Args:

    """
    logger = setup_logging(name="ML Model Predict Flow")
    logger.info("Starting ML predict flow")
    
    # Load data
    dataset = load_new_data()
    # Load model
    model, scaler = load_model()

    # Make predictions
    predictions = make_predictions(model=model, dataset=dataset, scaler=scaler)

    # Temp: print classification report
    from sklearn.metrics import classification_report
    print(classification_report(predictions["default payment next month"], predictions["predictions"]))
    
    # Train models
    # training_results = train_ml_models(
    #     dataset,
    #     use_hyperparameter_optimization=use_hyperparameter_optimization
    # )
    
    logger.info("ML prediction flow completed")

if __name__ == "__main__":
    ml_predict_flow()