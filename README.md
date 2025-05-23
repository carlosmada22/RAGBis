# openBIS Chatbot

A RAG-based chatbot for the openBIS documentation.

## Overview

This project provides a chatbot that can answer questions about openBIS using Retrieval Augmented Generation (RAG). It consists of three main components:

1. **Scraper**: Scrapes content from the openBIS documentation website
2. **Processor**: Processes the scraped content for use in RAG
3. **Query Engine**: Provides a chatbot interface for querying the processed content

## Installation

### Requirements

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) with the following models:
  - `nomic-embed-text` (for embeddings)
  - `qwen3` (for chat)

### From Source (Recommended)

```bash
git clone https://github.com/yourusername/openbis-chatbot.git
cd openbis-chatbot
pip install -e .
```

This installs the package in development mode, allowing you to make changes to the code and have them reflected immediately.

### Using pip (Not yet available)

```bash
pip install openbis-chatbot
```

Note: This option will be available once the package is published to PyPI.

## Usage

### Simple Usage (Recommended)

The simplest way to use the chatbot is with a single command:

```bash
python -m openbis_chatbot
```

This will:
1. Check if processed data already exists in the `data/processed` directory
2. If it exists, start the chatbot with that data
3. If not, automatically scrape the openBIS documentation, process it, and then start the chatbot

### Advanced Usage (Component-by-Component)

If you need more control, you can still run each component separately:

#### Scraping Content

```bash
python -m openbis_chatbot scrape --url https://openbis.readthedocs.io/en/latest/ --output ./data/raw
```

#### Processing Content

```bash
python -m openbis_chatbot process --input ./data/raw --output ./data/processed
```

#### Running the Chatbot (CLI)

```bash
python -m openbis_chatbot query --data ./data/processed
```

#### Running the Web Interface

```bash
python -m openbis_chatbot --web
```

This will start a web server on http://localhost:5000 where you can interact with the chatbot through a browser.

Alternatively, you can use the provided script:

```bash
python scripts/run_web.py
```

Or customize the web interface with additional parameters:

```bash
python -m openbis_chatbot.web.cli --data ./data/processed --host 127.0.0.1 --port 5000
```

## Command-Line Options

### Scraper

```
--url URL             The base URL of the ReadtheDocs site
--output OUTPUT       The directory to save the scraped content to
--version VERSION     The specific version to scrape (e.g., 'en/latest')
--delay DELAY         The delay between requests in seconds (default: 0.5)
--max-pages MAX_PAGES The maximum number of pages to scrape
--verbose             Enable verbose logging
```

### Processor

```
--input INPUT         The directory containing the scraped content
--output OUTPUT       The directory to save the processed content to
--min-chunk-size MIN_CHUNK_SIZE
                      The minimum size of a chunk in characters (default: 100)
--max-chunk-size MAX_CHUNK_SIZE
                      The maximum size of a chunk in characters (default: 1000)
--chunk-overlap CHUNK_OVERLAP
                      The overlap between chunks in characters (default: 50)
--verbose             Enable verbose logging
```

### Chatbot (CLI)

```
--data DATA           The directory containing the processed content
--model MODEL         The Ollama model to use for chat (default: qwen3)
--top-k TOP_K         The number of chunks to retrieve (default: 5)
--verbose             Enable verbose logging
```

### Web Interface

```
--data DATA           The directory containing the processed content (default: ./data/processed)
--host HOST           The host to run the web interface on (default: 0.0.0.0)
--port PORT           The port to run the web interface on (default: 5000)
--model MODEL         The Ollama model to use for chat (default: qwen3)
--top-k TOP_K         The number of chunks to retrieve (default: 5)
--debug               Enable debug mode
```

## How It Works

### Scraper

The scraper works by:
1. Starting from the base URL of the openBIS documentation site
2. Downloading the HTML content of each page
3. Extracting links to other pages on the same domain
4. Following those links to scrape more pages
5. Saving the content of each page to a text file

### Processor

The processor works by:
1. Reading the scraped content from text files
2. Chunking the content into smaller pieces
3. Generating embeddings for each chunk using Ollama's embedding model
4. Saving the chunks and their embeddings to JSON and CSV files

### Query Engine

The query engine works by:
1. Loading the processed chunks and their embeddings
2. Generating an embedding for the user's query
3. Finding the most relevant chunks based on cosine similarity
4. Creating a prompt with the query and the relevant chunks
5. Generating an answer using Ollama's chat model

### Web Interface

The web interface works by:
1. Starting a Flask web server
2. Serving a responsive HTML/CSS/JavaScript chat interface
3. Handling API requests from the frontend
4. Using the query engine to generate responses
5. Returning the responses to the frontend in JSON format

## License

This project is licensed under the MIT License - see the LICENSE file for details.
