# Usage Guide

This guide explains how to use the openBIS Chatbot.

## Installation

### Prerequisites

First, make sure you have Ollama installed and running. You can download it from [ollama.ai](https://ollama.ai/).

You'll need the following models:
- `nomic-embed-text` (for embeddings)
- `qwen3` (for chat)

You can pull these models with:
```bash
ollama pull nomic-embed-text
ollama pull qwen3
```

### Installation from Source (Recommended)

Clone the repository and install in development mode:

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

## Simple Usage (Recommended)

The simplest way to use the chatbot is with a single command:

```bash
python -m openbis_chatbot
```

This will:
1. Check if processed data already exists in the `data/processed` directory
2. If it exists, start the chatbot with that data
3. If not, automatically scrape the openBIS documentation, process it, and then start the chatbot

This approach requires no additional parameters and handles the entire pipeline automatically.

### Specifying a Different Model

By default, the chatbot uses the `qwen3` model from Ollama. If you want to use a different model, you can specify it with the `--model` parameter:

```bash
python -m openbis_chatbot --model llama3
```

Available models depend on your Ollama installation. Some common options include:
- `qwen3` (default)
- `llama3`
- `mistral`
- `gemma`

Make sure the model is available in your Ollama installation before using it.

## Advanced Usage (Component-by-Component)

If you need more control, you can still run each component separately:

### Scraping Content

To scrape content from the openBIS documentation website:

```bash
python -m openbis_chatbot scrape --url https://openbis.readthedocs.io/en/latest/ --output ./data/raw
```

This will download all the documentation pages and save them as text files in the `data/raw` directory.

### Processing Content

To process the scraped content for use in RAG:

```bash
python -m openbis_chatbot process --input ./data/raw --output ./data/processed
```

This will chunk the content, generate embeddings, and save the processed data in the `data/processed` directory.

### Running the Chatbot (CLI)

To run the chatbot with previously processed data in the command line:

```bash
python -m openbis_chatbot query --data ./data/processed
```

This will start an interactive command-line chatbot interface where you can ask questions about openBIS.

You can further customize the CLI interface with these options:
- `--top-k`: The number of chunks to retrieve (default: 5)
- `--verbose`: Enable verbose logging

### Running the Web Interface

To run the chatbot with a web interface, you have several options:

#### Option 1: Using the --web flag (Recommended)

```bash
python -m openbis_chatbot --web
```

This will start a web server on http://localhost:5000 where you can interact with the chatbot through a browser. The web interface provides a more user-friendly experience with a modern chat interface.

#### Option 2: Using the provided script

```bash
python scripts/run_web.py
```

This will also start a web server with default settings.

#### Option 3: Using the web module directly (for advanced customization)

```bash
python -m openbis_chatbot.web.cli --data ./data/processed --host 127.0.0.1 --port 5000
```

You can customize the web server with these options:
- `--host`: The host to run the web interface on (default: 0.0.0.0)
- `--port`: The port to run the web interface on (default: 5000)
- `--model`: The Ollama model to use for chat (default: qwen3)
- `--top-k`: The number of chunks to retrieve (default: 5)
- `--debug`: Enable debug mode for development

## Example Questions

Here are some example questions you can ask the chatbot:

- What is openBIS?
- How do I create a new collection in openBIS?
- How can I register a new object in openBIS?
- What are the main features of openBIS?
- How do I search for data in openBIS?
