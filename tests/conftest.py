"""
Pytest configuration file.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing the scraper."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page — openBIS Documentation</title>
    </head>
    <body>
        <div class="document">
            <h1>Test Page</h1>
            <p>This is a test paragraph.</p>
            <div class="sphinxsidebar">
                <p>This should be ignored.</p>
            </div>
            <a href="https://example.com/page1.html">Page 1</a>
            <a href="https://example.com/page2.html">Page 2</a>
            <a href="https://other-domain.com/page.html">External Page</a>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_text_content():
    """Sample text content for testing the processor."""
    return """Title: Test Page
URL: https://example.com/test
---

# Test Page

This is a test paragraph.

## Section 1

This is the first section.

## Section 2

This is the second section.
"""


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing the query engine."""
    return [
        {
            "title": "Test Page",
            "content": "This is a test paragraph.",
            "embedding": [0.1] * 768,
            "url": "https://example.com/test",
            "chunk_id": "test_0"
        },
        {
            "title": "Test Page",
            "content": "This is the first section.",
            "embedding": [0.2] * 768,
            "url": "https://example.com/test",
            "chunk_id": "test_1"
        },
        {
            "title": "Test Page",
            "content": "This is the second section.",
            "embedding": [0.3] * 768,
            "url": "https://example.com/test",
            "chunk_id": "test_2"
        }
    ]
