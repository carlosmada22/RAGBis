# RAGbis - Data Acquisition and Processing for openBIS Documentation

RAGbis is a standalone Python package that scrapes openBIS documentation, processes the content, and generates embeddings for use in RAG (Retrieval Augmented Generation) applications.

## Features

- **Web Scraping**: Automatically scrapes openBIS documentation from ReadtheDocs
- **Content Processing**: Intelligently chunks content while preserving document structure
- **Embedding Generation**: Creates embeddings using Ollama's `nomic-embed-text` model
- **Data Export**: Saves processed data in JSON and CSV formats for easy consumption

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- The `nomic-embed-text` model installed in Ollama

### Installing Ollama and Required Models

1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Pull the required embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```

## Installation

### From Source

1. Clone or download this project
2. Navigate to the RAGbis_project directory
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Using pip (if published)

```bash
pip install ragbis
```

## Usage

### Basic Usage

Run RAGbis with default settings to scrape and process openBIS documentation:

```bash
python -m ragbis
```

This will:
- Scrape the openBIS documentation from the default URL
- Save raw content to `./data/raw/`
- Process and generate embeddings
- Save processed data to `./data/processed/`

### Command Line Options

```bash
python -m ragbis --help
```

Available options:

- `--url URL`: Base URL to scrape (default: https://openbis.readthedocs.io/en/latest/)
- `--output-dir DIR`: Output directory for data (default: ./data)
- `--max-pages N`: Maximum number of pages to scrape (default: 100)
- `--delay SECONDS`: Delay between requests (default: 0.5)
- `--force-rebuild`: Force rebuild even if processed data exists
- `--min-chunk-size N`: Minimum chunk size in characters (default: 100)
- `--max-chunk-size N`: Maximum chunk size in characters (default: 1000)
- `--chunk-overlap N`: Chunk overlap in characters (default: 50)
- `--verbose`: Enable verbose logging

### Examples

Scrape with custom settings:
```bash
python -m ragbis --max-pages 200 --output-dir ./my_data --verbose
```

Force rebuild existing data:
```bash
python -m ragbis --force-rebuild
```

Custom chunking parameters:
```bash
python -m ragbis --min-chunk-size 200 --max-chunk-size 1500 --chunk-overlap 100
```

## Output Structure

RAGbis creates the following directory structure:

```
data/
├── raw/                    # Raw scraped content
│   ├── index.txt
│   ├── installation.txt
│   └── ...
└── processed/             # Processed data for RAG
    ├── chunks.json        # Main data file with embeddings
    └── chunks.csv         # Metadata without embeddings
```

### Output Files

- **chunks.json**: Contains all processed chunks with embeddings, titles, URLs, and content
- **chunks.csv**: Contains chunk metadata without embeddings for easy inspection

## Integration with chatBIS

The processed data from RAGbis is designed to be used with chatBIS, the conversational interface. After running RAGbis, you can:

1. Copy the `data` directory to your chatBIS project
2. Or point chatBIS to the RAGbis output directory

## Configuration

### Environment Variables

- `OLLAMA_HOST`: Ollama server host (default: localhost)
- `OLLAMA_PORT`: Ollama server port (default: 11434)

### Customizing the Scraper

You can modify the scraping behavior by editing the scraper configuration in the source code:

- Target different documentation versions
- Adjust content selectors for different site layouts
- Modify delay and retry settings

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check if the model is installed: `ollama list`
   - Install the model if missing: `ollama pull nomic-embed-text`

2. **Memory Issues**
   - Reduce `--max-pages` for large documentation sites
   - Increase `--min-chunk-size` to create fewer chunks
   - Process in smaller batches

3. **Network Issues**
   - Increase `--delay` between requests
   - Check your internet connection
   - Verify the documentation URL is accessible

### Logging

Enable verbose logging to debug issues:
```bash
python -m ragbis --verbose
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
```

### Type Checking

```bash
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Ensure Ollama is properly configured
