#!/usr/bin/env python3
"""
Run the openBIS Chatbot web interface.
"""

import logging
from openbis_chatbot.web.app import run_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting web interface...")
    run_app(debug=True)
