#!/usr/bin/env python3
"""
Main entry point for RAGbis - Data acquisition and processing for openBIS documentation.

This module orchestrates the scraping and processing workflow to generate
embeddings and processed data for RAG applications.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from ragbis.scraper.scraper import ReadTheDocsScraper
from ragbis.processor.processor import RAGProcessor
from ragbis.utils.logging import setup_logging

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_OPENBIS_URL = "https://openbis.readthedocs.io/en/latest/"
DEFAULT_MAX_PAGES = 100
DEFAULT_OUTPUT_DIR = "./data"


def check_processed_data_exists(output_dir: str) -> bool:
    """Check if processed data already exists."""
    processed_dir = Path(output_dir) / "processed"
    chunks_file = processed_dir / "chunks.json"
    return chunks_file.exists()


def run_scraper(url: str, raw_dir: str, max_pages: int = None, delay: float = 0.5) -> bool:
    """
    Run the scraper to collect raw documentation.
    
    Args:
        url: The base URL to scrape
        raw_dir: Directory to save raw scraped content
        max_pages: Maximum number of pages to scrape
        delay: Delay between requests
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Starting scraper for {url}")
        scraper = ReadTheDocsScraper(
            base_url=url,
            output_dir=raw_dir,
            max_pages=max_pages,
            delay=delay
        )
        scraper.scrape()
        logger.info("Scraping completed successfully")
        return True
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        return False


def run_processor(raw_dir: str, processed_dir: str, **kwargs) -> bool:
    """
    Run the processor to generate embeddings and processed data.
    
    Args:
        raw_dir: Directory containing raw scraped content
        processed_dir: Directory to save processed content
        **kwargs: Additional processor arguments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Starting processor for {raw_dir}")
        processor = RAGProcessor(
            input_dir=raw_dir,
            output_dir=processed_dir,
            **kwargs
        )
        processor.process()
        logger.info("Processing completed successfully")
        return True
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return False


def main():
    """Main entry point for RAGbis."""
    parser = argparse.ArgumentParser(
        description="RAGbis - Data acquisition and processing for openBIS documentation",
        prog="ragbis"
    )
    
    parser.add_argument(
        "--url", 
        default=DEFAULT_OPENBIS_URL,
        help=f"Base URL to scrape (default: {DEFAULT_OPENBIS_URL})"
    )
    parser.add_argument(
        "--output-dir", 
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for data (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--max-pages", 
        type=int, 
        default=DEFAULT_MAX_PAGES,
        help=f"Maximum number of pages to scrape (default: {DEFAULT_MAX_PAGES})"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--force-rebuild", 
        action="store_true",
        help="Force rebuild even if processed data exists"
    )
    parser.add_argument(
        "--min-chunk-size", 
        type=int, 
        default=100,
        help="Minimum chunk size in characters (default: 100)"
    )
    parser.add_argument(
        "--max-chunk-size", 
        type=int, 
        default=1000,
        help="Maximum chunk size in characters (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=50,
        help="Chunk overlap in characters (default: 50)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Create output directories
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    
    # Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"RAGbis starting with output directory: {output_dir}")
    
    # Check if processed data already exists
    if not args.force_rebuild and check_processed_data_exists(args.output_dir):
        logger.info("Processed data already exists. Use --force-rebuild to regenerate.")
        logger.info(f"Processed data location: {processed_dir}")
        return 0
    
    # Step 1: Run scraper
    logger.info("Step 1: Scraping documentation...")
    if not run_scraper(
        url=args.url,
        raw_dir=str(raw_dir),
        max_pages=args.max_pages,
        delay=args.delay
    ):
        logger.error("Scraping failed. Exiting.")
        return 1
    
    # Step 2: Run processor
    logger.info("Step 2: Processing and generating embeddings...")
    if not run_processor(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        min_chunk_size=args.min_chunk_size,
        max_chunk_size=args.max_chunk_size,
        chunk_overlap=args.chunk_overlap
    ):
        logger.error("Processing failed. Exiting.")
        return 1
    
    logger.info("RAGbis completed successfully!")
    logger.info(f"Raw data saved to: {raw_dir}")
    logger.info(f"Processed data saved to: {processed_dir}")
    logger.info("You can now use the processed data with chatBIS.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
