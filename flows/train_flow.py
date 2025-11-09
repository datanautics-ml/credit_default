# flows/train_flow.py
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

@task(name="Load Training Data ", retries=1)
def load_data(file_path: Path) -> pd.DataFrame:
    """Load training data from CSV file"""
    logger = get_run_logger()
    logger.info(f"Loading data from {file_path}")
    data = pd.read_csv(file_path)
    logger.info(f"Data shape: {data.shape}")
    return data

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
def ml_training_flow(
    data_file: Path = settings.DATA_DIR / "data_01.csv",
    use_hyperparameter_optimization: bool = True
) -> None:
    """
    Prefect flow to orchestrate ML model training
    
    Args:
        data_file: Path to the processed training data CSV
        use_hyperparameter_optimization: Whether to use hyperparameter optimization
    """
    logger = setup_logging(name="ML Model Training Flow")
    logger.info("Starting ML training flow")
    
    # Load data
    dataset = load_data(data_file)
    
    # Train models
    training_results = train_ml_models(
        dataset,
        use_hyperparameter_optimization=use_hyperparameter_optimization
    )
    
    logger.info("ML training flow completed")

if __name__ == "__main__":
    ml_training_flow(use_hyperparameter_optimization=True)