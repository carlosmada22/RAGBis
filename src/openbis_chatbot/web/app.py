#!/usr/bin/env python3
"""
Flask web application for the openBIS Chatbot.
"""

import os
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify

from openbis_chatbot.query.query import RAGQueryEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data", "processed")

app = Flask(__name__)

# Initialize the query engine
query_engine = None


def initialize_query_engine(data_dir=DEFAULT_DATA_DIR, model="qwen3"):
    """Initialize the query engine."""
    global query_engine
    try:
        logger.info(f"Initializing query engine with data from {data_dir}...")
        query_engine = RAGQueryEngine(
            data_dir=data_dir,
            api_key=None,
            model=model
        )
        logger.info("Query engine initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Error initializing query engine: {e}")
        return False


@app.route('/')
def index():
    """Render the chat interface."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    global query_engine

    # Check if the query engine is initialized
    if query_engine is None:
        success = initialize_query_engine()
        if not success:
            return jsonify({
                'success': False,
                'error': 'Failed to initialize query engine. Please check the logs.'
            })

    # Get the query from the request
    data = request.json
    query = data.get('query', '')

    if not query.strip():
        return jsonify({
            'success': False,
            'error': 'Query cannot be empty.'
        })

    try:
        # Query the processed content
        answer, relevant_chunks = query_engine.query(query, top_k=3)

        # Return the answer and sources
        return jsonify({
            'success': True,
            'answer': answer,
            'sources': [
                {'title': chunk['title'], 'url': chunk['url']}
                for chunk in relevant_chunks
            ]
        })

    except Exception as e:
        logger.error(f"Error querying: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })


def run_app(host='0.0.0.0', port=5000, debug=False, data_dir=DEFAULT_DATA_DIR, model="qwen3"):
    """Run the Flask application."""
    # Initialize the query engine
    initialize_query_engine(data_dir, model)

    # Run the Flask application
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app(debug=True)
