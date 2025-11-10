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
    data = pd.read_csv(latest_file_path)
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
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
@task(name="Train Models")
def train_ml_models(
    dataset: pd.DataFrame,
    use_hyperparameter_optimization: bool = True
) -> Dict[str, Any]:
    """
    Train multiple ML models
    
    Args:
        dataset: dataframe containing features and target
        use_hyperparameter_optimization: Whether to use hyperparameter optimization
        
    Returns:
        Training results for all models
    """
    logger = get_run_logger()
    logger.info("Starting ML model training")
    
    try:
        trainer = ModelTrainer()
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(dataset)

        # Train all models
        results = trainer.train_all_models(
            X_train, y_train, X_test, y_test,
            use_hyperparameter_optimization=use_hyperparameter_optimization
        )
        
        # Get best model
        best_model_name, best_model, best_metrics = trainer.get_best_model()
        
        # Save models
        models_dir = trainer.save_models()
        
        logger.info(f"Training completed. Best model: {best_model_name}")
        
        # Create model comparison table
        comparison_data = []
        for model_name, model_results in results.items():
            metrics = model_results['evaluation']
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        # Sort by AUC (higher is better)
        comparison_data.sort(key=lambda x: float(x['AUC']), reverse=True)
        
        create_table_artifact(
            key="model-comparison",
            table={
                'Model': [row['Model'] for row in comparison_data],
                'Accuracy': [row['Accuracy'] for row in comparison_data],
                'Recall': [row['Recall'] for row in comparison_data],
                'AUC': [row['AUC'] for row in comparison_data]
            },
            description="ML model performance comparison (sorted by AUC)"
        )
        
        # Create detailed results summary
        results_summary = f"""
# ML Training Results

## Best Performing Model
**{best_model_name.replace('_', ' ').title()}**
- **Accuracy**: {best_metrics['accuracy']:.4f} eV
- **Recall**: {best_metrics['recall']:.4f} eV
- **AUC**: {best_metrics['roc_auc']:.4f}

## All Models Performance
"""
        
        for i, row in enumerate(comparison_data, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
            results_summary += f"\n{emoji} **{row['Model']}**: Accuracy={row['Accuracy']}, Recall={row['Recall']}, AUC={row['AUC']}"
        
        results_summary += f"""

## Training Configuration
- **Hyperparameter Optimization**: {'✅ Enabled' if use_hyperparameter_optimization else '❌ Disabled'}
- **Cross-Validation Folds**: {settings.CV_FOLDS}
- **Models Trained**: {len(results)}
- **Models Saved**: `{models_dir}`

## MLflow Tracking
- **Tracking URI**: {settings.MLFLOW_TRACKING_URI}
- **Experiment**: {settings.MLFLOW_EXPERIMENT_NAME}
        """
        
        create_markdown_artifact(
            key="training-results-summary",
            markdown=results_summary,
            description="Comprehensive ML training results and model comparison"
        )
        
        return {
            'results': results,
            'best_model_name': best_model_name,
            'best_metrics': best_metrics,
            'models_saved_to': str(models_dir)
        }
        
    except Exception as e:
        logger.error(f"ML training failed: {e}")
        raise

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
    model = load_model()
    
    # Train models
    # training_results = train_ml_models(
    #     dataset,
    #     use_hyperparameter_optimization=use_hyperparameter_optimization
    # )
    
    logger.info("ML prediction flow completed")

if __name__ == "__main__":
    ml_predict_flow()