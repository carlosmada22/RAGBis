"""
Main entry point for the openBIS Chatbot package.
"""

import argparse
import sys

from openbis_chatbot.scraper.cli import main as scraper_main
from openbis_chatbot.processor.cli import main as processor_main
from openbis_chatbot.query.cli import main as query_main


def main():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(
        description="openBIS Chatbot - A RAG-based chatbot for the openBIS documentation.",
        prog="openbis-chatbot"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Scraper command
    scraper_parser = subparsers.add_parser("scrape", help="Scrape content from a ReadtheDocs site")
    scraper_parser.add_argument("--url", required=True, help="The base URL of the ReadtheDocs site")
    scraper_parser.add_argument("--output", required=True, help="The directory to save the scraped content to")
    scraper_parser.add_argument("--version", help="The specific version to scrape (e.g., 'en/latest')")
    scraper_parser.add_argument("--delay", type=float, default=0.5, help="The delay between requests in seconds")
    scraper_parser.add_argument("--max-pages", type=int, help="The maximum number of pages to scrape")
    scraper_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Processor command
    processor_parser = subparsers.add_parser("process", help="Process content for RAG")
    processor_parser.add_argument("--input", required=True, help="The directory containing the scraped content")
    processor_parser.add_argument("--output", required=True, help="The directory to save the processed content to")
    processor_parser.add_argument("--api-key", help="Not used for Ollama, kept for compatibility")
    processor_parser.add_argument("--min-chunk-size", type=int, default=100, help="The minimum size of a chunk in characters")
    processor_parser.add_argument("--max-chunk-size", type=int, default=1000, help="The maximum size of a chunk in characters")
    processor_parser.add_argument("--chunk-overlap", type=int, default=50, help="The overlap between chunks in characters")
    processor_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query processed content using RAG")
    query_parser.add_argument("--data", required=True, help="The directory containing the processed content")
    query_parser.add_argument("--api-key", help="Not used for Ollama, kept for compatibility")
    query_parser.add_argument("--model", default="qwen3", help="The Ollama model to use for chat")
    query_parser.add_argument("--top-k", type=int, default=3, help="The number of chunks to retrieve")
    query_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.command == "scrape":
        return scraper_main()
    elif args.command == "process":
        return processor_main()
    elif args.command == "query":
        return query_main()
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
