#!/usr/bin/env python3
"""
Check the lab notebook chunks.
"""

import json

def main():
    # Load the chunks
    with open('data/processed/chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Find lab notebook chunks
    lab_notebook_chunks = [chunk for chunk in chunks if 'Lab Notebook' in chunk['title']]
    
    # Print the number of lab notebook chunks
    print(f"Found {len(lab_notebook_chunks)} lab notebook chunks")
    
    # Print the content of each lab notebook chunk
    for i, chunk in enumerate(lab_notebook_chunks, 1):
        print(f"\nChunk {i}:")
        print(f"Title: {chunk['title']}")
        print(f"URL: {chunk['url']}")
        print(f"Content: {chunk['content']}")
        print("-" * 80)
    
    # Check if any chunk contains "Register a Collection"
    register_collection_chunks = [chunk for chunk in chunks if "Register a Collection" in chunk['content']]
    
    # Print the number of chunks containing "Register a Collection"
    print(f"\nFound {len(register_collection_chunks)} chunks containing 'Register a Collection'")
    
    # Print the content of each chunk containing "Register a Collection"
    for i, chunk in enumerate(register_collection_chunks, 1):
        print(f"\nChunk {i}:")
        print(f"Title: {chunk['title']}")
        print(f"URL: {chunk['url']}")
        print(f"Content: {chunk['content']}")
        print("-" * 80)

if __name__ == "__main__":
    main()
