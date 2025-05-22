"""
Unit tests for the Ollama utilities.
"""

from unittest.mock import patch, MagicMock

from src.openbis_chatbot.utils.ollama import check_ollama_availability


class TestOllama:
    """Tests for the Ollama utilities."""

    @patch("langchain_ollama.OllamaEmbeddings")
    def test_check_ollama_availability_success(self, mock_ollama_embeddings):
        """Test checking Ollama availability when it's available."""
        # Mock the OllamaEmbeddings class
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 768
        mock_ollama_embeddings.return_value = mock_embeddings

        # Check Ollama availability
        available, message = check_ollama_availability()

        # Check that OllamaEmbeddings was initialized correctly
        mock_ollama_embeddings.assert_called_once_with(model="nomic-embed-text")

        # Check that embed_query was called
        mock_embeddings.embed_query.assert_called_once_with("test")

        # Check that Ollama is reported as available
        assert available is True
        assert message == "Ollama server is running and embeddings are working."

    @patch("langchain_ollama.OllamaEmbeddings")
    def test_check_ollama_availability_empty_result(self, mock_ollama_embeddings):
        """Test checking Ollama availability when it returns an empty result."""
        # Mock the OllamaEmbeddings class
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = None
        mock_ollama_embeddings.return_value = mock_embeddings

        # Check Ollama availability
        available, message = check_ollama_availability()

        # Check that OllamaEmbeddings was initialized correctly
        mock_ollama_embeddings.assert_called_once_with(model="nomic-embed-text")

        # Check that embed_query was called
        mock_embeddings.embed_query.assert_called_once_with("test")

        # Check that Ollama is reported as unavailable
        assert available is False
        assert message == "Ollama embeddings returned empty result."

    @patch("langchain_ollama.OllamaEmbeddings")
    def test_check_ollama_availability_error(self, mock_ollama_embeddings):
        """Test checking Ollama availability when it raises an error."""
        # Mock the OllamaEmbeddings class
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.side_effect = Exception("Test error")
        mock_ollama_embeddings.return_value = mock_embeddings

        # Check Ollama availability
        available, message = check_ollama_availability()

        # Check that OllamaEmbeddings was initialized correctly
        mock_ollama_embeddings.assert_called_once_with(model="nomic-embed-text")

        # Check that embed_query was called
        mock_embeddings.embed_query.assert_called_once_with("test")

        # Check that Ollama is reported as unavailable
        assert available is False
        assert message == "Error connecting to Ollama server: Test error"

    @patch("builtins.__import__")
    def test_check_ollama_availability_import_error(self, mock_import):
        """Test checking Ollama availability when the import fails."""
        # Configure the mock to raise ImportError when importing langchain_ollama
        def import_side_effect(name, *args, **kwargs):
            if name == 'langchain_ollama':
                raise ImportError("Test error")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = import_side_effect

        # Check Ollama availability
        available, message = check_ollama_availability()

        # Check that Ollama is reported as unavailable
        assert available is False
        assert message == "Langchain Ollama package not available."
