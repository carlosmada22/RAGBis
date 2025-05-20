#!/usr/bin/env python3
"""
Setup script for the openBIS Chatbot package.
"""

from setuptools import setup, find_packages

setup(
    name="openbis-chatbot",
    version="0.1.0",
    description="A RAG-based chatbot for the openBIS documentation",
    author="AI Assistant",
    author_email="example@example.com",
    url="https://github.com/yourusername/openbis-chatbot",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    scripts=[
        "scripts/openbis-scraper",
        "scripts/openbis-processor",
        "scripts/openbis-chatbot",
    ],
    install_requires=[
        "beautifulsoup4>=4.12.0",
        "langchain-core>=0.1.0",
        "langchain-ollama>=0.0.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "scikit-learn>=1.3.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
