#!/usr/bin/env python3
"""
RAG Query Engine for ReadtheDocs Content

This module provides functionality for querying the processed content using RAG
(Retrieval Augmented Generation).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logger = logging.getLogger(__name__)

# Try to import Ollama, but don't fail if it's not available
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    OLLAMA_AVAILABLE = True
    # Check if Ollama server is running
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        # Test with a simple embedding
        test_embedding = embeddings.embed_query("test")
        if test_embedding:
            logger.info("Ollama server is running and embeddings are working.")
        else:
            logger.warning("Ollama embeddings returned empty result.")
            OLLAMA_AVAILABLE = False
    except Exception as e:
        logger.warning(f"Error connecting to Ollama server: {e}")
        OLLAMA_AVAILABLE = False
except ImportError:
    logger.warning("Langchain Ollama package not available. Using dummy embeddings.")
    OLLAMA_AVAILABLE = False


class RAGQueryEngine:
    """Class for querying processed content using RAG."""

    def __init__(self, data_dir: str, api_key: Optional[str] = None, model: str = "qwen3"):
        """
        Initialize the RAG query engine.
        
        Args:
            data_dir: The directory containing the processed content
            api_key: Not used for Ollama, kept for compatibility
            model: The Ollama model to use for chat
        """
        self.data_dir = Path(data_dir)
        self.api_key = api_key  # Not used for Ollama, kept for compatibility
        self.model = model
        
        # Load the processed chunks
        self.chunks = self._load_chunks()
        
        if OLLAMA_AVAILABLE:
            logger.info("Using Ollama for embeddings and completions")
            self.embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
            self.llm = ChatOllama(model=self.model)
        else:
            logger.warning("Ollama not available or not running")
            self.embeddings_model = None
            self.llm = None
    
    def _load_chunks(self) -> List[Dict]:
        """
        Load the processed chunks from the data directory.
        
        Returns:
            A list of dictionaries containing the processed chunks
        """
        chunks_file = self.data_dir / "chunks.json"
        
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
        
        return chunks
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a text.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            The embedding for the text
        """
        if OLLAMA_AVAILABLE and self.embeddings_model:
            try:
                # Use Ollama's embedding API
                embedding = self.embeddings_model.embed_query(text)
                return embedding
            
            except Exception as e:
                logger.error(f"Error generating embedding with Ollama: {e}")
                logger.warning("Falling back to dummy embedding")
        
        # If Ollama is not available or there was an error, use a dummy embedding
        return self._generate_dummy_embedding()
    
    def _generate_dummy_embedding(self, dim: int = 1536) -> List[float]:
        """
        Generate a dummy embedding (random vector).
        
        Args:
            dim: The dimension of the embedding
            
        Returns:
            A random vector of the specified dimension
        """
        # Generate a random vector
        embedding = np.random.normal(0, 1, dim)
        
        # Normalize it to unit length
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query.
        
        Args:
            query: The query to retrieve chunks for
            top_k: The number of chunks to retrieve
            
        Returns:
            A list of the most relevant chunks
        """
        # Generate an embedding for the query
        query_embedding = self.generate_embedding(query)
        
        # Calculate the similarity between the query and each chunk
        similarities = []
        for chunk in self.chunks:
            chunk_embedding = chunk["embedding"]
            similarity = cosine_similarity([query_embedding], [chunk_embedding])[0][0]
            similarities.append((chunk, similarity))
        
        # Sort the chunks by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top_k chunks
        return [chunk for chunk, _ in similarities[:top_k]]
    
    def generate_answer(self, query: str, relevant_chunks: List[Dict]) -> str:
        """
        Generate an answer for a query using the relevant chunks.
        
        Args:
            query: The query to generate an answer for
            relevant_chunks: The relevant chunks to use for generating the answer
            
        Returns:
            The generated answer
        """
        if not OLLAMA_AVAILABLE or not self.llm:
            return "Ollama not available or not running. Cannot generate answer."
        
        try:
            # Create a prompt for the language model
            prompt = self._create_prompt(query, relevant_chunks)
            
            # Create the system instruction and user prompt
            system_instruction = "You are a helpful assistant that answers questions based on the provided documentation."
            
            # Generate the response using Ollama
            full_prompt = system_instruction + "\n\n" + prompt
            response = self.llm.invoke(full_prompt)
            
            # Extract the answer from the response
            answer = response.content
            
            return answer
        
        except Exception as e:
            logger.error(f"Error generating answer with Ollama: {e}")
            return f"Error generating answer: {e}"
    
    def _create_prompt(self, query: str, relevant_chunks: List[Dict]) -> str:
        """
        Create a prompt for the language model.
        
        Args:
            query: The query to create a prompt for
            relevant_chunks: The relevant chunks to include in the prompt
            
        Returns:
            The prompt for the language model
        """
        # Create a prompt with the query and the relevant chunks
        prompt = f"Question: {query}\n\n"
        prompt += "Here is some relevant documentation that might help answer the question:\n\n"
        
        for i, chunk in enumerate(relevant_chunks, 1):
            prompt += f"Document {i}:\n"
            prompt += f"Title: {chunk['title']}\n"
            prompt += f"Content: {chunk['content']}\n\n"
        
        prompt += """Based on the above documentation, please answer the question in a conversational and helpful way.
If the documentation doesn't contain the answer, respond in a friendly manner and ask if there's something else you can help with.
Do not mention the sources or documents in your answer.
If the user is just saying hello or making small talk, respond in a friendly way as an assistant specialized in openBIS without requiring specific documentation references.
Remember that you are an AI assistant specializing in openBIS, a system for managing research data."""
        
        return prompt
    
    def query(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        """
        Query the processed content using RAG.
        
        Args:
            query: The query to answer
            top_k: The number of chunks to retrieve
            
        Returns:
            A tuple containing the answer and the relevant chunks
        """
        logger.info(f"Query: {query}")
        
        # Retrieve the most relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=top_k)
        
        # Generate an answer
        answer = self.generate_answer(query, relevant_chunks)
        
        return answer, relevant_chunks
