"""
Unit tests for Settings configuration
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.settings import Settings


class TestSettings:
    """Test suite for Settings class"""

    def test_settings_initialization(self):
        """Test that Settings initializes with default values"""
        settings = Settings()

        assert settings.PROJECT_ROOT is not None
        assert isinstance(settings.DATA_DIR, Path)
        assert isinstance(settings.MODELS_DIR, Path)
        assert isinstance(settings.LOGS_DIR, Path)
        assert settings.TEST_SIZE == 0.2
        assert settings.RANDOM_STATE == 42
        assert settings.CV_FOLDS == 5

    def test_settings_paths_exist(self):
        """Test that Settings creates required paths"""
        settings = Settings()

        # Paths should be Path objects
        assert isinstance(settings.DATA_DIR, Path)
        assert isinstance(settings.RAW_DATA_DIR, Path)
        assert isinstance(settings.PROCESSED_DATA_DIR, Path)
        assert isinstance(settings.MODELS_DIR, Path)
        assert isinstance(settings.LOGS_DIR, Path)

    def test_settings_environment_variables(self):
        """Test that Settings can read from environment variables"""
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://test:5000"}):
            settings = Settings()
            assert settings.MLFLOW_TRACKING_URI == "http://test:5000"

    def test_settings_default_values(self):
        """Test default configuration values"""
        settings = Settings()

        assert settings.MLFLOW_TRACKING_URI == "http://localhost:5000"
        assert settings.MLFLOW_EXPERIMENT_NAME == "materials_bandgap_prediction"
        assert settings.PREFECT_API_URL == "http://localhost:4200/api"
        assert settings.TEST_SIZE == 0.2
        assert settings.RANDOM_STATE == 42
        assert settings.CV_FOLDS == 5
        assert settings.N_TRIALS == 100

    def test_settings_path_relationships(self):
        """Test that path relationships are correct"""
        settings = Settings()

        assert settings.RAW_DATA_DIR.parent == settings.DATA_DIR
        assert settings.PROCESSED_DATA_DIR.parent == settings.DATA_DIR
        assert settings.FEATURES_DATA_DIR.parent == settings.DATA_DIR
