"""
Unit tests for the ContentChunker class.
"""

import pytest
from src.openbis_chatbot.processor.processor import ContentChunker


class TestContentChunker:
    """Tests for the ContentChunker class."""

    def test_init(self):
        """Test initialization of the chunker."""
        chunker = ContentChunker(
            min_chunk_size=100,
            max_chunk_size=1000,
            chunk_overlap=50
        )
        
        # Check that the attributes were initialized correctly
        assert chunker.min_chunk_size == 100
        assert chunker.max_chunk_size == 1000
        assert chunker.chunk_overlap == 50

    def test_chunk_content_empty(self):
        """Test chunking empty content."""
        chunker = ContentChunker()
        
        # Test with empty string
        chunks = chunker.chunk_content("")
        assert chunks == []
        
        # Test with whitespace
        chunks = chunker.chunk_content("   \n\n   ")
        assert chunks == []

    def test_chunk_content_single_chunk(self):
        """Test chunking content that fits in a single chunk."""
        chunker = ContentChunker(
            min_chunk_size=10,
            max_chunk_size=1000,
            chunk_overlap=0
        )
        
        content = "This is a test paragraph.\n\nThis is another paragraph."
        chunks = chunker.chunk_content(content)
        
        # Check that we got a single chunk
        assert len(chunks) == 1
        
        # Check that the chunk contains all the content
        assert "This is a test paragraph." in chunks[0]
        assert "This is another paragraph." in chunks[0]

    def test_chunk_content_multiple_chunks(self):
        """Test chunking content into multiple chunks."""
        chunker = ContentChunker(
            min_chunk_size=10,
            max_chunk_size=30,
            chunk_overlap=0
        )
        
        content = "This is paragraph 1.\n\nThis is paragraph 2.\n\nThis is paragraph 3."
        chunks = chunker.chunk_content(content)
        
        # Check that we got multiple chunks
        assert len(chunks) > 1
        
        # Check that each chunk contains the expected content
        assert "This is paragraph 1." in chunks[0]
        assert "This is paragraph 3." in chunks[-1]

    def test_chunk_content_with_overlap(self):
        """Test chunking content with overlap."""
        chunker = ContentChunker(
            min_chunk_size=10,
            max_chunk_size=30,
            chunk_overlap=10
        )
        
        content = "This is paragraph 1.\n\nThis is paragraph 2.\n\nThis is paragraph 3."
        chunks = chunker.chunk_content(content)
        
        # Check that we got multiple chunks
        assert len(chunks) > 1
        
        # Check that there is overlap between chunks
        # The exact overlap depends on the implementation
        # but we should see some text from the first chunk in the second
        overlap_text = chunks[0][-10:]
        assert overlap_text in chunks[1]

    def test_chunk_content_with_min_chunk_size(self):
        """Test chunking content with minimum chunk size constraint."""
        chunker = ContentChunker(
            min_chunk_size=50,
            max_chunk_size=100,
            chunk_overlap=0
        )
        
        # Create content with small paragraphs
        content = "\n\n".join(["Small paragraph {}".format(i) for i in range(10)])
        
        chunks = chunker.chunk_content(content)
        
        # Check that each chunk meets the minimum size
        for chunk in chunks:
            assert len(chunk) >= chunker.min_chunk_size

    def test_chunk_content_with_large_paragraph(self):
        """Test chunking content with a paragraph larger than max_chunk_size."""
        chunker = ContentChunker(
            min_chunk_size=10,
            max_chunk_size=50,
            chunk_overlap=0
        )
        
        # Create content with a large paragraph
        large_paragraph = "This is a very large paragraph that exceeds the maximum chunk size."
        content = "Small paragraph.\n\n" + large_paragraph + "\n\nAnother small paragraph."
        
        chunks = chunker.chunk_content(content)
        
        # Check that we got multiple chunks
        assert len(chunks) > 1
        
        # Check that the large paragraph is in one of the chunks
        assert any(large_paragraph in chunk for chunk in chunks)
