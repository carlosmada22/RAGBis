# Usage Guide

This guide explains how to use the openBIS Chatbot.

## Installation

First, make sure you have Ollama installed and running. You can download it from [ollama.ai](https://ollama.ai/).

Then, install the openBIS Chatbot package:

```bash
pip install openbis-chatbot
```

## Scraping Content

To scrape content from the openBIS documentation website, use the `openbis-scraper` command:

```bash
openbis-scraper --url https://openbis.readthedocs.io/en/latest/ --output ./scraped_content --version en/latest
```

This will download all the documentation pages and save them as text files in the `scraped_content` directory.

## Processing Content

To process the scraped content for use in RAG, use the `openbis-processor` command:

```bash
openbis-processor --input ./scraped_content --output ./processed_content
```

This will chunk the content, generate embeddings, and save the processed data in the `processed_content` directory.

## Running the Chatbot

To run the chatbot, use the `openbis-chatbot` command:

```bash
openbis-chatbot --data ./processed_content
```

This will start an interactive chatbot interface where you can ask questions about openBIS.

## Example Questions

Here are some example questions you can ask the chatbot:

- What is openBIS?
- How do I create a new collection in openBIS?
- How can I register a new object in openBIS?
- What are the main features of openBIS?
- How do I search for data in openBIS?
