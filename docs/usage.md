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

### Running the Chatbot

To run the chatbot with previously processed data:

```bash
python -m openbis_chatbot query --data ./data/processed
```

This will start an interactive chatbot interface where you can ask questions about openBIS.

## Example Questions

Here are some example questions you can ask the chatbot:

- What is openBIS?
- How do I create a new collection in openBIS?
- How can I register a new object in openBIS?
- What are the main features of openBIS?
- How do I search for data in openBIS?
