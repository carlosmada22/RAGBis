#!/usr/bin/env python3
"""
Test script to debug the query engine specifically for collection registration.
"""

import json
import logging
from openbis_chatbot.query.query import RAGQueryEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    # Initialize the query engine
    query_engine = RAGQueryEngine(
        data_dir="data/processed",
        model="qwen3"
    )
    
    # Test query
    query = "How to register a collection in openBIS"
    
    # Retrieve relevant chunks
    relevant_chunks = query_engine.retrieve_relevant_chunks(query, top_k=10)
    
    # Print the relevant chunks
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(relevant_chunks)} chunks:\n")
    
    # Check if any chunk contains information about registering a collection
    collection_chunks = []
    for chunk in relevant_chunks:
        if "Register a Collection" in chunk['content']:
            collection_chunks.append(chunk)
    
    print(f"Found {len(collection_chunks)} chunks containing 'Register a Collection'")
    
    # Print the content of each chunk containing "Register a Collection"
    for i, chunk in enumerate(collection_chunks, 1):
        print(f"\nChunk {i}:")
        print(f"Title: {chunk['title']}")
        print(f"URL: {chunk['url']}")
        print(f"Content: {chunk['content']}")
        print("-" * 80)
    
    # Generate answer
    answer, _ = query_engine.query(query)
    
    # Print the answer
    print(f"\nGenerated Answer:\n{answer}")

if __name__ == "__main__":
    main()
