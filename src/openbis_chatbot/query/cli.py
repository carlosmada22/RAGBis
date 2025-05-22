#!/usr/bin/env python3
"""
Command-line interface for the RAG query engine.
"""

import argparse
import logging
import sys
from dotenv import load_dotenv

from openbis_chatbot.query.query import RAGQueryEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def parse_args(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Query processed content using RAG.")
    parser.add_argument("--data", required=True, help="The directory containing the processed content")
    parser.add_argument("--api-key", help="Not used for Ollama, kept for compatibility")
    parser.add_argument("--model", default="qwen3", help="The Ollama model to use for chat")
    parser.add_argument("--top-k", type=int, default=5, help="The number of chunks to retrieve")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args(args)


def run_with_args(args):
    """Run the query engine with the given arguments."""
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # API key is not used for Ollama, but kept for compatibility
    api_key = args.api_key

    try:
        # Create the query engine
        query_engine = RAGQueryEngine(
            data_dir=args.data,
            api_key=api_key,
            model=args.model
        )

        # Interactive query loop
        print("🤖 openBIS Assistant")
        print("I'm here to help you with questions about openBIS. Type 'exit' or 'quit' to end our conversation.")
        print()

        while True:
            # Get the query from the user
            query = input("You: ")

            # Exit if the user types 'exit' or 'quit'
            if query.lower() in ["exit", "quit"]:
                print("Goodbye! Have a great day!")
                break

            # Skip empty queries
            if not query.strip():
                continue

            try:
                # Query the processed content
                answer, relevant_chunks = query_engine.query(query, top_k=args.top_k)

                # Print the answer without the sources
                print(f"\nAssistant: {answer}\n")

                # Log the sources but don't display them
                if args.verbose:
                    logger.info("Sources used:")
                    for i, chunk in enumerate(relevant_chunks, 1):
                        logger.info(f"{i}. {chunk['title']} - {chunk['url']}")

            except Exception as e:
                logger.error(f"Error querying: {e}")
                print("I'm sorry, I encountered an error. Please try again or ask a different question.")

        return 0

    except Exception as e:
        logger.error(f"Error initializing query engine: {e}")
        return 1


def main():
    """Main entry point for the script."""
    args = parse_args()
    return run_with_args(args)


if __name__ == "__main__":
    sys.exit(main())
