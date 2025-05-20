"""
Unit tests for the logging utilities.
"""

import pytest
import logging
import sys
from unittest.mock import patch, MagicMock

from src.openbis_chatbot.utils.logging import setup_logging


class TestLogging:
    """Tests for the logging utilities."""

    @patch("logging.basicConfig")
    def test_setup_logging_default(self, mock_basicConfig):
        """Test setting up logging with default level."""
        # Call the setup_logging function
        logger = setup_logging()

        # Check that basicConfig was called with the correct arguments
        mock_basicConfig.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout
        )

        # Check that the logger was returned
        assert isinstance(logger, logging.Logger)

        # Check that the logger has the correct name
        assert logger.name == "src.openbis_chatbot.utils.logging"

    @patch("logging.basicConfig")
    def test_setup_logging_custom_level(self, mock_basicConfig):
        """Test setting up logging with a custom level."""
        # Call the setup_logging function with a custom level
        logger = setup_logging(level=logging.DEBUG)

        # Check that basicConfig was called with the correct arguments
        mock_basicConfig.assert_called_once_with(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout
        )

        # Check that the logger was returned
        assert isinstance(logger, logging.Logger)

        # Check that the logger has the correct name
        assert logger.name == "src.openbis_chatbot.utils.logging"

    @patch("logging.getLogger")
    def test_setup_logging_reduces_other_loggers(self, mock_getLogger):
        """Test that setup_logging reduces the level of other loggers."""
        # Mock the getLogger function to return a mock logger
        mock_urllib3_logger = MagicMock()
        mock_requests_logger = MagicMock()
        mock_httpx_logger = MagicMock()

        mock_getLogger.side_effect = lambda name: {
            "urllib3": mock_urllib3_logger,
            "requests": mock_requests_logger,
            "httpx": mock_httpx_logger,
            "__name__": MagicMock(),
            "src.openbis_chatbot.utils.logging": MagicMock()
        }[name]

        # Call the setup_logging function
        setup_logging()

        # Check that the level of the other loggers was set to WARNING
        mock_urllib3_logger.setLevel.assert_called_once_with(logging.WARNING)
        mock_requests_logger.setLevel.assert_called_once_with(logging.WARNING)
        mock_httpx_logger.setLevel.assert_called_once_with(logging.WARNING)
