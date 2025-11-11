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
from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently import Dataset
from evidently import DataDefinition
from evidently import BinaryClassification
import datetime as dt

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
    return data, latest_file

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
def make_predictions(model: Any, dataset: pd.DataFrame, latest_file: str, scaler: Any = None) -> pd.DataFrame:
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
     
     # Copy the datset with predictions to the processed data directory
    try:        
        settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = settings.PROCESSED_DATA_DIR / latest_file
        if not dest_path.exists():
            dataset.to_csv(dest_path, index=False)
            logger.info(f"Wrote {latest_file} to {dest_path}")
            if (settings.PROCESSED_DATA_DIR / "processed_data.csv").exists():
                processed_data = pd.read_csv(settings.PROCESSED_DATA_DIR / "processed_data.csv")
                processed_data = pd.concat([processed_data, dataset], ignore_index=True)
                processed_data.to_csv(settings.PROCESSED_DATA_DIR / "processed_data.csv", index=False)
            else:
                dataset.to_csv(settings.PROCESSED_DATA_DIR / "processed_data.csv", index=False)
            logger.info(f"Updated processed_data.csv in {settings.PROCESSED_DATA_DIR}")
        else:
            logger.info(f"Processed file already exists at {dest_path}")
    except Exception as e:
        logger.error(f"Failed to copy file to processed dir: {e}")
        raise
    logger.info("Predictions completed")
    return dataset

@task(name="Evaluate and Monior")
def evaluate_and_monitor(predictions: pd.DataFrame, reference_path="data/processed/processed_data.csv") -> None:
    """Generate data and performance drift reports"""
    logger = get_run_logger()
    logger.info("Generating data drift and performance reports")

    if Path(reference_path).exists():
        reference_data = pd.read_csv(reference_path)
    else:
        reference_data = predictions
    # reference_data.rename(columns={"default credit next month": "target"}, inplace=True)
    # predictions.rename(columns={"default credit next month": "target"}, inplace=True)

    # create evidently definition for binary classification
    definition = DataDefinition(
    classification=[BinaryClassification(
        target="default payment next month",
        prediction_labels="predictions")],
        id_column="ID",
        categorical_columns=["default payment next month", "predictions", 
                             "SEX","EDUCATION","MARRIAGE"
        ]
    )
    cols = [col for col in predictions.columns if col not in ["MONTH"]]#, "ID"]]
    # create evidently datasets
    current_data = Dataset.from_pandas(
        predictions[cols],
        data_definition=definition
    )
    reference_data = Dataset.from_pandas(
        reference_data[cols],
        data_definition=definition
    )

    
    
    # Create evidently reports
    report = Report(metrics=[
        DataDriftPreset(),
        ClassificationPreset()
    ])
    # Run the report
    eval = report.run(reference_data=reference_data, current_data=current_data)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    report_path = f"monitoring/reports/drift_report_{timestamp}.html"    
    eval.save_html(report_path)
    print(f"Drift report saved: {report_path}")     
    logger.info("Data drift and performance reports generated")
    return report_path

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
    dataset, latest_file = load_new_data()
    # Load model
    model, scaler = load_model()

    # Make predictions
    predictions = make_predictions(model=model, dataset=dataset,
                                   latest_file=latest_file, scaler=scaler)
    
    # Evaluate and monitor
    report_path = evaluate_and_monitor(predictions=predictions)

    # Temp: print classification report
    # print(classification_report(predictions["default payment next month"], predictions["predictions"]))
    
    # Train models
    # training_results = train_ml_models(
    #     dataset,
    #     use_hyperparameter_optimization=use_hyperparameter_optimization
    # )
    
    logger.info("ML prediction flow completed")

if __name__ == "__main__":
    ml_predict_flow()