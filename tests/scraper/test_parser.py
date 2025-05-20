"""
Unit tests for the ReadTheDocsParser class.
"""

from bs4 import BeautifulSoup
from src.openbis_chatbot.scraper.scraper import ReadTheDocsParser


class TestReadTheDocsParser:
    """Tests for the ReadTheDocsParser class."""

    def test_init(self):
        """Test initialization of the parser."""
        parser = ReadTheDocsParser()

        # Check that the content selectors were initialized correctly
        assert "div.document" in parser.content_selectors
        assert "div[role='main']" in parser.content_selectors
        assert "main" in parser.content_selectors
        assert "article" in parser.content_selectors
        assert "div.body" in parser.content_selectors

        # Check that the ignore selectors were initialized correctly
        assert "div.sphinxsidebar" in parser.ignore_selectors
        assert "footer" in parser.ignore_selectors
        assert "nav" in parser.ignore_selectors
        assert "div.header" in parser.ignore_selectors
        assert "div.related" in parser.ignore_selectors
        assert "div.breadcrumbs" in parser.ignore_selectors
        assert "div.sourcelink" in parser.ignore_selectors
        assert "div.highlight-default" in parser.ignore_selectors
        assert "div.admonition" in parser.ignore_selectors

    def test_extract_content_with_valid_html(self):
        """Test extracting content from valid HTML."""
        parser = ReadTheDocsParser()

        # Create a simple HTML document
        html_content = """
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
                </div>
            </body>
        </html>
        """

        result = parser.extract_content(html_content, "https://example.com/test")

        # Check that the title was extracted correctly
        assert result["title"] == "Test Page"

        # Check that the content was extracted correctly
        assert "This is a test paragraph" in result["content"]

        # Check that the ignored content was not included
        assert "This should be ignored" not in result["content"]

        # Check that the URL was preserved
        assert result["url"] == "https://example.com/test"

    def test_extract_content_with_no_content_element(self):
        """Test extracting content when no content element is found."""
        parser = ReadTheDocsParser()

        # Create a simple HTML document with no content element
        html_content = """
        <html>
            <head>
                <title>Test Page</title>
            </head>
            <body>
                <p>This is a test paragraph.</p>
            </body>
        </html>
        """

        result = parser.extract_content(html_content, "https://example.com/test")

        # Check that the title was extracted correctly
        assert result["title"] == "Test Page"

        # Check that the content is empty
        assert result["content"] == ""

        # Check that the URL was preserved
        assert result["url"] == "https://example.com/test"

    def test_extract_content_with_no_title(self):
        """Test extracting content when no title is found."""
        parser = ReadTheDocsParser()

        # Create a simple HTML document with no title
        html_content = """
        <html>
            <head>
            </head>
            <body>
                <div class="document">
                    <p>This is a test paragraph.</p>
                </div>
            </body>
        </html>
        """

        result = parser.extract_content(html_content, "https://example.com/test")

        # Check that the title is empty
        assert result["title"] == ""

        # Check that the content was extracted correctly
        assert "This is a test paragraph" in result["content"]

        # Check that the URL was preserved
        assert result["url"] == "https://example.com/test"

    def test_extract_text_with_structure_code(self):
        """Test extracting text from a code element."""
        parser = ReadTheDocsParser()

        # Create a code element
        soup = BeautifulSoup('<code>def test(): return "Hello"</code>', "html.parser")
        code_element = soup.code

        result = parser._extract_text_with_structure(code_element)

        # Check that the code was formatted correctly
        assert result.startswith("\n```\n")
        assert result.endswith("\n```\n")
        assert 'def test(): return "Hello"' in result

    def test_extract_text_with_structure_heading(self):
        """Test extracting text from a heading element."""
        parser = ReadTheDocsParser()

        # Create heading elements
        soup = BeautifulSoup('<h1>Title</h1><h2>Subtitle</h2>', "html.parser")
        h1_element = soup.h1
        h2_element = soup.h2

        h1_result = parser._extract_text_with_structure(h1_element)
        h2_result = parser._extract_text_with_structure(h2_element)

        # Check that the headings were formatted correctly
        assert h1_result.strip() == "# Title"
        assert h2_result.strip() == "## Subtitle"

    def test_extract_text_with_structure_paragraph(self):
        """Test extracting text from a paragraph element."""
        parser = ReadTheDocsParser()

        # Create a paragraph element
        soup = BeautifulSoup('<p>This is a paragraph.</p>', "html.parser")
        p_element = soup.p

        result = parser._extract_text_with_structure(p_element)

        # Check that the paragraph was formatted correctly
        assert result.strip() == "This is a paragraph."

    def test_extract_text_with_structure_list_item(self):
        """Test extracting text from a list item element."""
        parser = ReadTheDocsParser()

        # Create a list item element
        soup = BeautifulSoup('<li>List item</li>', "html.parser")
        li_element = soup.li

        result = parser._extract_text_with_structure(li_element)

        # Check that the list item was formatted correctly
        assert result.strip() == "- List item"

    def test_extract_text_with_structure_table(self):
        """Test extracting text from a table element."""
        parser = ReadTheDocsParser()

        # Create a table element
        soup = BeautifulSoup('<table><tr><td>Cell 1</td><td>Cell 2</td></tr></table>', "html.parser")
        table_element = soup.table

        result = parser._extract_text_with_structure(table_element)

        # Check that the table was formatted correctly
        assert "Cell 1" in result
        assert "Cell 2" in result

    def test_extract_text_with_structure_nested(self):
        """Test extracting text from a nested element."""
        parser = ReadTheDocsParser()

        # Create a nested element
        soup = BeautifulSoup('<div><h1>Title</h1><p>Paragraph</p></div>', "html.parser")
        div_element = soup.div

        result = parser._extract_text_with_structure(div_element)

        # Check that the nested elements were formatted correctly
        assert "\n# Title" in result
        assert "\nParagraph" in result
