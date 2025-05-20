"""
Unit tests for the main module.
"""

from unittest.mock import patch, MagicMock

from openbis_chatbot.__main__ import main


class TestMain:
    """Tests for the main module."""

    def test_main_no_command(self):
        """Test the main function with no command."""
        with patch("sys.argv", ["openbis-chatbot"]):
            with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
                with patch("argparse.ArgumentParser.print_help") as mock_print_help:
                    # Mock the argument parser to return a namespace with command=None
                    mock_args = MagicMock()
                    mock_args.command = None
                    mock_parse_args.return_value = mock_args

                    # Call the main function
                    result = main()

                    # Check that the help was printed
                    mock_print_help.assert_called_once()

                    # Check that the function returned 1 (error)
                    assert result == 1

    def test_main_invalid_command(self):
        """Test the main function with an invalid command."""
        with patch("sys.argv", ["openbis-chatbot", "invalid"]):
            with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
                with patch("argparse.ArgumentParser.print_help") as mock_print_help:
                    # Mock the argument parser to return a namespace with command="invalid"
                    mock_args = MagicMock()
                    mock_args.command = "invalid"
                    mock_parse_args.return_value = mock_args

                    # Call the main function
                    result = main()

                    # Check that the help was printed
                    mock_print_help.assert_called_once()

                    # Check that the function returned 1 (error)
                    assert result == 1
