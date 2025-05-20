"""
Tests for the ContentChunker class.
"""

import unittest
from openbis_chatbot.processor.processor import ContentChunker


class TestContentChunker(unittest.TestCase):
    """Tests for the ContentChunker class."""

    def test_chunk_content(self):
        """Test chunking content."""
        chunker = ContentChunker(
            min_chunk_size=10,
            max_chunk_size=50,
            chunk_overlap=5
        )
        
        content = """
        This is a test paragraph.
        
        This is another test paragraph that is a bit longer than the first one.
        
        And this is a third paragraph that should be in a separate chunk.
        
        Finally, this is a fourth paragraph.
        """
        
        chunks = chunker.chunk_content(content)
        
        # Check that we have the expected number of chunks
        self.assertEqual(len(chunks), 2)
        
        # Check that the chunks have the expected content
        self.assertIn("This is a test paragraph", chunks[0])
        self.assertIn("This is another test paragraph", chunks[0])
        self.assertIn("And this is a third paragraph", chunks[1])
        self.assertIn("Finally, this is a fourth paragraph", chunks[1])


if __name__ == "__main__":
    unittest.main()
