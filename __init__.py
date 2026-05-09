"""
Neuro Research Engine - Advanced AI-Powered Research Analysis System

A hyper-technical, production-grade research engine for qualitative and quantitative
data analysis using OpenAI LLMs with multi-agent architecture.

Features:
- Multi-agent analysis pipeline (Qualitative, Quantitative, Synthesis)
- Advanced NLP with spaCy and NLTK integration
- Statistical analysis with scipy and scikit-learn
- Async processing with aiohttp
- Rich CLI output with detailed reporting
- Multiple data source ingestion (URLs, files, APIs)
- Comprehensive validation with Pydantic
- Integration hooks for webhooks, Slack, Discord

Usage:
    python main.py --url "https://example.com" --output analysis
    python main.py --file data.csv --mode quantitative --output stats
    python main.py --text "Raw text data" --mode qualitative --output themes

Author: Neuro Research Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Neuro Research Team"
__license__ = "MIT"
