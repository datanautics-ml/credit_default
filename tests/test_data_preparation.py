"""
Unit tests for data preparation functions
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.models.training import ModelTrainer


class TestDataPreparation:
    """Test suite for data preparation"""

    @pytest.fixture
    def trainer(self, temp_dir, data_dir_with_files):
        """Create a ModelTrainer instance for testing"""
        with patch("src.models.training.settings") as mock_settings, patch("src.models.training.mlflow"):
            mock_settings.MLFLOW_TRACKING_URI = "sqlite:///test.db"
            mock_settings.MODELS_DIR = temp_dir / "models"
            mock_settings.DATA_DIR = data_dir_with_files
            mock_settings.TEST_SIZE = 0.2
            mock_settings.RANDOM_STATE = 42
            mock_settings.CV_FOLDS = 3

            trainer = ModelTrainer(experiment_name="test")
            return trainer

    def test_prepare_data_removes_id_column(self, trainer, sample_data):
        """Test that ID column is removed during data preparation"""
        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_data)
        
        assert "ID" not in X_train.columns
        assert "ID" not in X_test.columns

    def test_prepare_data_removes_target_column(self, trainer, sample_data):
        """Test that target column is removed from features"""
        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_data)
        
        assert "default payment next month" not in X_train.columns
        assert "default payment next month" not in X_test.columns

    def test_prepare_data_maintains_feature_columns(self, trainer, sample_data):
        """Test that feature columns are preserved"""
        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_data)
        
        # Check that feature columns are present
        assert "LIMIT_BAL" in X_train.columns
        assert "LIMIT_BAL" in X_test.columns
        assert "AGE" in X_train.columns
        assert "AGE" in X_test.columns

    def test_prepare_data_correct_split_ratio(self, trainer, sample_data):
        """Test that train-test split maintains correct ratio"""
        X_train, X_test, y_train, y_test = trainer.prepare_data(sample_data)
        
        total_samples = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total_samples
        
        # Allow small tolerance for rounding
        assert abs(test_ratio - 0.2) < 0.05

    def test_prepare_data_reproducible_split(self, trainer, sample_data):
        """Test that data split is reproducible with same random state"""
        X_train1, X_test1, y_train1, y_test1 = trainer.prepare_data(sample_data)
        X_train2, X_test2, y_train2, y_test2 = trainer.prepare_data(sample_data)
        
        # Check that splits are identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)
