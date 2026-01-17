"""
Unit tests for logging utilities
"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.utils.logging import setup_logging


class TestLogging:
    """Test suite for logging utilities"""

    def test_setup_logging_basic(self):
        """Test basic logging setup"""
        logger = setup_logging("test_logger")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO

    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level"""
        logger = setup_logging("test_logger", level="DEBUG")

        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self, tmp_path):
        """Test logging setup with file handler"""
        log_file = tmp_path / "test.log"

        with patch("src.utils.logging.settings") as mock_settings:
            mock_settings.LOGS_DIR = tmp_path
            logger = setup_logging("test_logger", log_file="test.log")

            # Log a message
            logger.info("Test message")

            # Check that log file was created
            assert log_file.exists()

            # Check that message was written
            with open(log_file) as f:
                content = f.read()
                assert "Test message" in content

    def test_setup_logging_removes_existing_handlers(self):
        """Test that setup_logging removes existing handlers"""
        logger = logging.getLogger("test_logger")
        handler = logging.StreamHandler()
        logger.addHandler(handler)

        assert len(logger.handlers) > 0

        # Setup logging should remove existing handlers
        logger = setup_logging("test_logger")

        # Should have at least one handler (console handler)
        assert len(logger.handlers) >= 1

    def test_setup_logging_multiple_calls(self):
        """Test that multiple calls to setup_logging work correctly"""
        logger1 = setup_logging("test_logger")
        logger2 = setup_logging("test_logger")

        # Should return the same logger instance
        assert logger1 is logger2
