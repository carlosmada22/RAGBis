"""
Unit tests for the processor CLI.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock

from src.openbis_chatbot.processor.cli import main


class TestProcessorCLI:
    """Tests for the processor CLI."""

    @patch("src.openbis_chatbot.processor.cli.RAGProcessor")
    @patch("sys.argv", ["openbis-processor", "--input", "./input", "--output", "./output", "--verbose"])
    def test_main_success(self, mock_processor_class):
        """Test the main function with valid arguments."""
        # Mock the processor instance
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        # Call the main function
        result = main()
        
        # Check that the processor was created with the correct arguments
        mock_processor_class.assert_called_once_with(
            input_dir="./input",
            output_dir="./output",
            api_key=None,
            min_chunk_size=100,
            max_chunk_size=1000,
            chunk_overlap=50
        )
        
        # Check that the process method was called
        mock_processor.process.assert_called_once()
        
        # Check that the function returned 0 (success)
        assert result == 0

    @patch("src.openbis_chatbot.processor.cli.RAGProcessor")
    @patch("sys.argv", ["openbis-processor", "--input", "./input", "--output", "./output", 
                       "--api-key", "test_key", "--min-chunk-size", "50", 
                       "--max-chunk-size", "500", "--chunk-overlap", "25"])
    def test_main_with_all_arguments(self, mock_processor_class):
        """Test the main function with all arguments."""
        # Mock the processor instance
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        # Call the main function
        result = main()
        
        # Check that the processor was created with the correct arguments
        mock_processor_class.assert_called_once_with(
            input_dir="./input",
            output_dir="./output",
            api_key="test_key",
            min_chunk_size=50,
            max_chunk_size=500,
            chunk_overlap=25
        )
        
        # Check that the process method was called
        mock_processor.process.assert_called_once()
        
        # Check that the function returned 0 (success)
        assert result == 0

    @patch("src.openbis_chatbot.processor.cli.RAGProcessor")
    @patch("sys.argv", ["openbis-processor", "--input", "./input", "--output", "./output"])
    def test_main_error(self, mock_processor_class):
        """Test the main function when an error occurs."""
        # Mock the processor instance
        mock_processor = MagicMock()
        mock_processor.process.side_effect = Exception("Test error")
        mock_processor_class.return_value = mock_processor
        
        # Call the main function
        result = main()
        
        # Check that the function returned 1 (error)
        assert result == 1
