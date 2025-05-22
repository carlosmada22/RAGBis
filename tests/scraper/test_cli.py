"""
Unit tests for the scraper CLI.
"""


from unittest.mock import patch, MagicMock

from src.openbis_chatbot.scraper.cli import main


class TestScraperCLI:
    """Tests for the scraper CLI."""

    @patch("src.openbis_chatbot.scraper.cli.ReadTheDocsScraper")
    @patch("sys.argv", ["openbis-scraper", "--url", "https://example.com", "--output", "./output", "--verbose"])
    def test_main_success(self, mock_scraper_class):
        """Test the main function with valid arguments."""
        # Mock the scraper instance
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper

        # Call the main function
        result = main()

        # Check that the scraper was created with the correct arguments
        mock_scraper_class.assert_called_once_with(
            base_url="https://example.com",
            output_dir="./output",
            target_version=None,
            delay=0.5,
            max_pages=None
        )

        # Check that the scrape method was called
        mock_scraper.scrape.assert_called_once()

        # Check that the function returned 0 (success)
        assert result == 0

    @patch("src.openbis_chatbot.scraper.cli.ReadTheDocsScraper")
    @patch("sys.argv", ["openbis-scraper", "--url", "https://example.com", "--output", "./output",
                       "--version", "en/latest", "--delay", "1.0", "--max-pages", "10"])
    def test_main_with_all_arguments(self, mock_scraper_class):
        """Test the main function with all arguments."""
        # Mock the scraper instance
        mock_scraper = MagicMock()
        mock_scraper_class.return_value = mock_scraper

        # Call the main function
        result = main()

        # Check that the scraper was created with the correct arguments
        mock_scraper_class.assert_called_once_with(
            base_url="https://example.com",
            output_dir="./output",
            target_version="en/latest",
            delay=1.0,
            max_pages=10
        )

        # Check that the scrape method was called
        mock_scraper.scrape.assert_called_once()

        # Check that the function returned 0 (success)
        assert result == 0

    @patch("src.openbis_chatbot.scraper.cli.ReadTheDocsScraper")
    @patch("sys.argv", ["openbis-scraper", "--url", "https://example.com", "--output", "./output"])
    def test_main_error(self, mock_scraper_class):
        """Test the main function when an error occurs."""
        # Mock the scraper instance
        mock_scraper = MagicMock()
        mock_scraper.scrape.side_effect = Exception("Test error")
        mock_scraper_class.return_value = mock_scraper

        # Call the main function
        result = main()

        # Check that the function returned 1 (error)
        assert result == 1
