"""
Unit tests for ModelTrainer class
"""

import json
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.training import ModelTrainer

# Suppress convergence warnings for test data
warnings.filterwarnings("ignore", category=UserWarning, message=".*convergence.*")
warnings.filterwarnings("ignore", message=".*did not converge.*")
warnings.filterwarnings("ignore", message=".*Maximum number of iterations.*")


class TestModelTrainer:
    """Test suite for ModelTrainer class"""

    @pytest.fixture
    def trainer(self, temp_dir, data_dir_with_files):
        """Create a ModelTrainer instance with mocked dependencies"""
        with patch("src.models.training.settings") as mock_settings:
            mock_settings.MLFLOW_TRACKING_URI = "sqlite:///test.db"
            mock_settings.MODELS_DIR = temp_dir / "models"
            mock_settings.DATA_DIR = data_dir_with_files
            mock_settings.TEST_SIZE = 0.2
            mock_settings.RANDOM_STATE = 42
            mock_settings.CV_FOLDS = 3  # Reduced for faster tests

            # Mock MLflow to avoid actual tracking
            with patch("src.models.training.mlflow"), warnings.catch_warnings():
                # Suppress convergence warnings for test data (expected with random data)
                warnings.filterwarnings("ignore", category=UserWarning, message=".*convergence.*")
                warnings.filterwarnings("ignore", message=".*did not converge.*")
                warnings.filterwarnings("ignore", message=".*Maximum number of iterations.*")
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                trainer = ModelTrainer(experiment_name="test_experiment")
                yield trainer

    def test_trainer_initialization(self, trainer):
        """Test that ModelTrainer initializes correctly"""
        assert trainer is not None
        assert trainer.experiment_name == "test_experiment"
        assert isinstance(trainer.models, dict)
        assert len(trainer.models) > 0
        assert isinstance(trainer.scalers, dict)
        assert len(trainer.scalers) > 0

    def test_trainer_data_setup(self, temp_dir, data_dir_with_files):
        """Test that ModelTrainer can initialize with data files present"""
        with patch("src.models.training.settings") as mock_settings:
            mock_settings.MLFLOW_TRACKING_URI = "sqlite:///test.db"
            mock_settings.MODELS_DIR = temp_dir / "models"
            mock_settings.DATA_DIR = data_dir_with_files
            mock_settings.TEST_SIZE = 0.2
            mock_settings.RANDOM_STATE = 42
            mock_settings.CV_FOLDS = 3

            # Verify data files exist
            expected_files = [f"data_{i:02d}.csv" for i in range(1, 13)]
            actual_files = set(f.name for f in data_dir_with_files.iterdir() if f.is_file())
            for expected_file in expected_files:
                assert expected_file in actual_files, f"Missing expected file: {expected_file}"

            # Mock MLflow to avoid actual tracking
            with patch("src.models.training.mlflow"):
                # This should not raise an error now that files exist
                trainer = ModelTrainer(experiment_name="test_experiment")
                assert trainer is not None

    def test_setup_models(self, trainer):
        """Test that models are set up correctly"""
        models = trainer._setup_models()

        assert isinstance(models, dict)
        assert "logistic_regression" in models
        assert "random_forest" in models
        assert "xgboost" in models

        # Check model structure
        for model_name, model_config in models.items():
            assert "model" in model_config
            assert "params" in model_config

    def test_setup_scalers(self, trainer):
        """Test that scalers are set up correctly"""
        scalers = trainer._setup_scalers()

        assert isinstance(scalers, dict)
        assert "standard" in scalers
        assert "robust" in scalers

    def test_prepare_data(self, trainer, sample_data):
        """Test data preparation and train-test split"""
        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_data)

        # Check shapes
        assert X_train.shape[0] + X_test.shape[0] == len(sample_data)
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

        # Check that target column is removed from features
        assert "default payment next month" not in X_train.columns
        assert "default payment next month" not in X_test.columns

        # Check that ID column is removed
        assert "ID" not in X_train.columns
        assert "ID" not in X_test.columns

    @patch("src.models.training.mlflow")
    def test_train_model_without_optimization(self, mock_mlflow, trainer, sample_train_test_data):
        """Test training a model without hyperparameter optimization"""
        X_train, X_test, y_train, y_test = sample_train_test_data

        # Mock MLflow context manager
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)

        # Suppress convergence warnings (expected with random test data)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*convergence.*")
            warnings.filterwarnings("ignore", message=".*did not converge.*")
            warnings.filterwarnings("ignore", message=".*Maximum number of iterations.*")
            result = trainer.train_model("logistic_regression", X_train, y_train, use_hyperparameter_optimization=False)

        assert "model" in result
        assert "best_params" in result
        assert "cv_score" in result
        assert result["model"] is not None
        assert "logistic_regression" in trainer.trained_models

    @patch("src.models.training.mlflow")
    def test_train_model_with_optimization(self, mock_mlflow, trainer, sample_train_test_data):
        """Test training a model with hyperparameter optimization"""
        X_train, X_test, y_train, y_test = sample_train_test_data

        # Mock MLflow
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)

        # Suppress convergence warnings (expected with random test data)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*convergence.*")
            warnings.filterwarnings("ignore", message=".*did not converge.*")
            warnings.filterwarnings("ignore", message=".*Maximum number of iterations.*")
            result = trainer.train_model("ridge", X_train, y_train, use_hyperparameter_optimization=True)

        assert "model" in result
        assert "best_params" in result
        assert result["model"] is not None
        assert "ridge" in trainer.trained_models

    def test_evaluate_model(self, trainer, sample_train_test_data):
        """Test model evaluation"""
        X_train, X_test, y_train, y_test = sample_train_test_data

        # Train a simple model first
        with patch("src.models.training.mlflow"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*convergence.*")
            warnings.filterwarnings("ignore", message=".*did not converge.*")
            warnings.filterwarnings("ignore", message=".*Maximum number of iterations.*")
            mock_mlflow = MagicMock()
            mock_mlflow.start_run.return_value.__enter__ = MagicMock()
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)

            trainer.train_model("logistic_regression", X_train, y_train, use_hyperparameter_optimization=False)

        # Evaluate the model
        with patch("src.models.training.mlflow"):
            metrics = trainer.evaluate_model("logistic_regression", X_test, y_test)

        # Check metrics structure
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        # Check metric values are valid
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_evaluate_model_not_trained(self, trainer, sample_train_test_data):
        """Test that evaluating an untrained model raises an error"""
        X_train, X_test, y_train, y_test = sample_train_test_data

        with pytest.raises(ValueError, match="not trained yet"):
            trainer.evaluate_model("logistic_regression", X_test, y_test)

    @patch("src.models.training.mlflow")
    def test_train_all_models(self, mock_mlflow, trainer, sample_train_test_data):
        """Test training all models"""
        X_train, X_test, y_train, y_test = sample_train_test_data

        # Mock MLflow
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)

        # Suppress convergence warnings (expected with random test data)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*convergence.*")
            warnings.filterwarnings("ignore", message=".*did not converge.*")
            warnings.filterwarnings("ignore", message=".*Maximum number of iterations.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            results = trainer.train_all_models(X_train, y_train, X_test, y_test, use_hyperparameter_optimization=False)

        assert isinstance(results, dict)
        assert len(results) > 0

        # Check result structure
        for model_name, model_results in results.items():
            assert "training" in model_results
            assert "evaluation" in model_results
            assert "accuracy" in model_results["evaluation"]

    @patch("src.models.training.mlflow")
    def test_get_best_model(self, mock_mlflow, trainer, sample_train_test_data):
        """Test getting the best model"""
        X_train, X_test, y_train, y_test = sample_train_test_data

        # Mock MLflow
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)

        # Train all models (suppress convergence warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*convergence.*")
            warnings.filterwarnings("ignore", message=".*did not converge.*")
            warnings.filterwarnings("ignore", message=".*Maximum number of iterations.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            trainer.train_all_models(X_train, y_train, X_test, y_test, use_hyperparameter_optimization=False)

        # Get best model
        best_name, best_model, best_metrics = trainer.get_best_model()

        assert best_name is not None
        assert best_model is not None
        assert isinstance(best_metrics, dict)
        assert "f1" in best_metrics

    def test_get_best_model_no_training(self, trainer):
        """Test that getting best model without training raises an error"""
        with pytest.raises(ValueError, match="No models trained yet"):
            trainer.get_best_model()

    @patch("src.models.training.mlflow")
    def test_save_models(self, mock_mlflow, trainer, sample_train_test_data, temp_dir):
        """Test saving models to disk"""
        X_train, X_test, y_train, y_test = sample_train_test_data

        # Mock MLflow
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)

        # Train a model (suppress convergence warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*convergence.*")
            warnings.filterwarnings("ignore", message=".*did not converge.*")
            warnings.filterwarnings("ignore", message=".*Maximum number of iterations.*")
            training_result = trainer.train_model(
                "logistic_regression", X_train, y_train, use_hyperparameter_optimization=False
            )
            evaluation_result = trainer.evaluate_model("logistic_regression", X_test, y_test)

            # Manually populate self.results (train_all_models does this automatically)
            # This is needed because save_models reads from self.results
            trainer.results["logistic_regression"] = {"training": training_result, "evaluation": evaluation_result}

        # Save models
        output_dir = trainer.save_models(temp_dir / "models")

        # Check that files were created
        assert (output_dir / "logistic_regression_model.joblib").exists()
        assert (output_dir / "standard_scaler.joblib").exists()
        assert (output_dir / "training_results.json").exists()

        # Check JSON structure
        with open(output_dir / "training_results.json") as f:
            results = json.load(f)
            assert "logistic_regression" in results

    def test_prepare_data_handles_missing_target(self, trainer, sample_data):
        """Test that prepare_data handles missing target column gracefully"""
        data_without_target = sample_data.drop(columns=["default payment next month"])

        with pytest.raises(KeyError):
            trainer.prepare_data(data_without_target)

    def test_prepare_data_handles_empty_dataframe(self, trainer):
        """Test that prepare_data handles empty dataframe"""
        empty_data = pd.DataFrame()

        with pytest.raises((ValueError, KeyError)):
            trainer.prepare_data(empty_data)
