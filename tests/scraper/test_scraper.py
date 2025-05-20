"""
Unit tests for the ReadTheDocsScraper class.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import requests
from bs4 import BeautifulSoup

from src.openbis_chatbot.scraper.scraper import ReadTheDocsScraper, ReadTheDocsParser


class TestReadTheDocsScraper:
    """Tests for the ReadTheDocsScraper class."""

    def test_init(self):
        """Test initialization of the scraper."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = ReadTheDocsScraper(
                base_url="https://example.com",
                output_dir=temp_dir,
                target_version="en/latest",
                delay=0.5,
                max_pages=10
            )

            # Check that the attributes were initialized correctly
            assert scraper.base_url == "https://example.com/"
            assert scraper.output_dir == Path(temp_dir)
            assert scraper.target_version == "en/latest"
            assert scraper.delay == 0.5
            assert scraper.max_pages == 10
            assert scraper.domain == "example.com"
            assert scraper.visited_urls == set()
            assert scraper.urls_to_visit == ["https://example.com/"]
            assert isinstance(scraper.parser, ReadTheDocsParser)

            # Check that the output directory was created
            assert os.path.exists(temp_dir)

    def test_sanitize_url(self):
        """Test sanitizing URLs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = ReadTheDocsScraper(
                base_url="example.com",
                output_dir=temp_dir
            )

            # Test adding https:// to a URL
            assert scraper._sanitize_url("example.com") == "https://example.com/"

            # Test adding a trailing slash to a URL
            assert scraper._sanitize_url("https://example.com") == "https://example.com/"

            # Test a URL that already has https:// and a trailing slash
            assert scraper._sanitize_url("https://example.com/") == "https://example.com/"

    def test_is_valid_url(self):
        """Test checking if a URL is valid for scraping."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = ReadTheDocsScraper(
                base_url="https://example.com",
                output_dir=temp_dir,
                target_version="en/latest"
            )

            # Add a URL to the visited set
            scraper.visited_urls.add("https://example.com/visited")

            # Test a valid URL
            # The URL should contain the domain and not be in the visited set
            # and not have a file extension that should be skipped
            # and match the target version if specified
            valid_url = "https://example.com/en/latest/page.html"
            assert scraper._is_valid_url(valid_url) is True

            # Test a URL on a different domain
            assert scraper._is_valid_url("https://other-domain.com/page.html") is False

            # Test a URL with a file extension that should be skipped
            assert scraper._is_valid_url("https://example.com/image.png") is False

            # Test a URL with an anchor
            # The URL should contain the domain and not be in the visited set
            # and not have a file extension that should be skipped
            # and match the target version if specified
            anchor_url = "https://example.com/en/latest/page.html#section"
            assert scraper._is_valid_url(anchor_url) is True

            # Test a URL that has already been visited
            assert scraper._is_valid_url("https://example.com/visited") is False

            # Test a URL that doesn't match the target version
            assert scraper._is_valid_url("https://example.com/en/stable/page.html") is False

    def test_save_content(self):
        """Test saving content to a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = ReadTheDocsScraper(
                base_url="https://example.com",
                output_dir=temp_dir
            )

            content = {
                "title": "Test Page",
                "content": "This is test content.",
                "url": "https://example.com/test-page.html"
            }

            # Mock the open function to avoid actually writing to a file
            with patch("builtins.open", mock_open()) as mock_file:
                scraper._save_content(content)

                # Check that the file was opened with the correct path and mode
                expected_path = os.path.join(temp_dir, "test-page.html.txt")
                mock_file.assert_called_once_with(Path(expected_path), "w", encoding="utf-8")

                # Check that the correct content was written to the file
                handle = mock_file()
                handle.write.assert_any_call("Title: Test Page\n")
                handle.write.assert_any_call("URL: https://example.com/test-page.html\n")
                handle.write.assert_any_call("---\n\n")
                handle.write.assert_any_call("This is test content.")

    def test_save_content_with_index_url(self):
        """Test saving content from an index URL."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = ReadTheDocsScraper(
                base_url="https://example.com",
                output_dir=temp_dir
            )

            content = {
                "title": "Index Page",
                "content": "This is the index page.",
                "url": "https://example.com/"
            }

            # Mock the open function to avoid actually writing to a file
            with patch("builtins.open", mock_open()) as mock_file:
                scraper._save_content(content)

                # Check that the file was opened with the correct path and mode
                expected_path = os.path.join(temp_dir, "index.txt")
                mock_file.assert_called_once_with(Path(expected_path), "w", encoding="utf-8")

    @patch("requests.get")
    @patch("time.sleep")  # Mock sleep to speed up the test
    def test_scrape(self, mock_sleep, mock_get):
        """Test scraping a site."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a scraper with a max of 2 pages
            scraper = ReadTheDocsScraper(
                base_url="https://example.com",
                output_dir=temp_dir,
                max_pages=2
            )

            # Mock the parser's extract_content method
            scraper.parser.extract_content = MagicMock(return_value={
                "title": "Test Page",
                "content": "This is test content.",
                "url": "https://example.com/page.html"
            })

            # Mock the _save_content method
            scraper._save_content = MagicMock()

            # Mock the requests.get response
            mock_response = MagicMock()
            mock_response.text = """
            <html>
                <body>
                    <a href="https://example.com/page1.html">Page 1</a>
                    <a href="https://example.com/page2.html">Page 2</a>
                    <a href="https://other-domain.com/page.html">External Page</a>
                </body>
            </html>
            """
            mock_get.return_value = mock_response

            # Call the scrape method
            scraper.scrape()

            # Check that requests.get was called twice (for the base URL and page1.html)
            assert mock_get.call_count == 2

            # Check that _save_content was called twice
            assert scraper._save_content.call_count == 2

            # Check that the visited_urls set contains the expected URLs
            assert "https://example.com/" in scraper.visited_urls
            assert "https://example.com/page1.html" in scraper.visited_urls

            # Check that the urls_to_visit list contains the expected URL
            assert "https://example.com/page2.html" in scraper.urls_to_visit

    @patch("requests.get")
    def test_scrape_with_error(self, mock_get):
        """Test scraping when an error occurs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a scraper
            scraper = ReadTheDocsScraper(
                base_url="https://example.com",
                output_dir=temp_dir
            )

            # Mock the requests.get method to raise an exception
            mock_get.side_effect = requests.exceptions.RequestException("Test error")

            # Call the scrape method
            scraper.scrape()

            # Check that the visited_urls set is empty
            assert len(scraper.visited_urls) == 0
