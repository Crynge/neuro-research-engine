"""
Data ingestion module with support for multiple sources.
Handles URL fetching, file parsing, and raw text processing.
"""

import os
import csv
import json
import aiohttp
import asyncio
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse

from config import get_settings


class DataIngestionError(Exception):
    """Custom exception for data ingestion errors."""
    pass


class URLFetcher:
    """
    Async URL content fetcher with retry logic and proper headers.
    Supports HTML parsing and text extraction.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.headers = {
            "User-Agent": self.settings.default_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    
    async def fetch(self, url: str, max_retries: int = None) -> str:
        """
        Fetch content from URL with retry logic.
        
        Args:
            url: Target URL to fetch
            max_retries: Maximum retry attempts (uses config default if None)
            
        Returns:
            Extracted text content from the page
            
        Raises:
            DataIngestionError: If fetching fails after all retries
        """
        max_retries = max_retries or self.settings.max_retries
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession(headers=self.headers) as session:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=self.settings.request_timeout),
                        allow_redirects=True
                    ) as response:
                        response.raise_for_status()
                        html = await response.text()
                        return self._extract_text_from_html(html, url)
                        
            except aiohttp.ClientError as e:
                last_error = f"HTTP error: {str(e)}"
            except asyncio.TimeoutError:
                last_error = f"Request timeout after {self.settings.request_timeout}s"
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
            
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise DataIngestionError(f"Failed to fetch {url} after {max_retries + 1} attempts: {last_error}")
    
    def _extract_text_from_html(self, html: str, url: str) -> str:
        """Extract clean text from HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Extract structured content
        title = soup.title.string if soup.title else ""
        meta_desc = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'})
        if meta_tag and meta_tag.get('content'):
            meta_desc = meta_tag['content']
        
        # Get main content
        text_elements = []
        
        # Prioritize main content areas
        main = soup.find('main') or soup.find('article') or soup.find('div', class_=lambda x: x and 'content' in x.lower())
        
        if main:
            text_elements.extend(main.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td']))
        else:
            text_elements.extend(soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
        
        # Build text content
        content_parts = []
        if title:
            content_parts.append(f"Title: {title.strip()}")
        if meta_desc:
            content_parts.append(f"Description: {meta_desc.strip()}")
        
        for elem in text_elements:
            text = elem.get_text(strip=True)
            if text and len(text) > 10:  # Filter very short fragments
                content_parts.append(text)
        
        full_text = "\n\n".join(content_parts)
        
        # Add source metadata
        result = f"Source URL: {url}\n\n{full_text}"
        return result[:500000]  # Limit to ~500k chars to avoid token limits


class FileParser:
    """
    Multi-format file parser supporting CSV, JSON, TXT, and Excel files.
    """
    
    SUPPORTED_EXTENSIONS = {'.csv', '.json', '.txt', '.xlsx', '.xls', '.md'}
    
    @classmethod
    def parse(cls, file_path: str) -> str:
        """
        Parse file and extract text content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text content
            
        Raises:
            DataIngestionError: If file cannot be parsed
        """
        path = Path(file_path)
        
        if not path.exists():
            raise DataIngestionError(f"File not found: {file_path}")
        
        if path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
            raise DataIngestionError(f"Unsupported file type: {path.suffix}")
        
        try:
            if path.suffix.lower() == '.csv':
                return cls._parse_csv(path)
            elif path.suffix.lower() == '.json':
                return cls._parse_json(path)
            elif path.suffix.lower() in {'.txt', '.md'}:
                return cls._parse_text(path)
            elif path.suffix.lower() in {'.xlsx', '.xls'}:
                return cls._parse_excel(path)
            else:
                raise DataIngestionError(f"Cannot parse file: {file_path}")
                
        except Exception as e:
            raise DataIngestionError(f"Error parsing {file_path}: {str(e)}")
    
    @staticmethod
    def _parse_csv(path: Path) -> str:
        """Parse CSV file and convert to structured text."""
        df = pd.read_csv(path)
        
        # Generate summary statistics for numeric columns
        summaries = []
        summaries.append(f"CSV File: {path.name}")
        summaries.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        summaries.append(f"Columns: {', '.join(df.columns.tolist())}")
        summaries.append("")
        
        # Add numeric summaries
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            summaries.append(f"{col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}, std={df[col].std():.2f}")
        
        # Add sample rows
        summaries.append("\nSample Data:")
        summaries.append(df.head(10).to_string())
        
        return "\n".join(summaries)
    
    @staticmethod
    def _parse_json(path: Path) -> str:
        """Parse JSON file and convert to readable text."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def flatten_json(obj, prefix=""):
            result = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    result.extend(flatten_json(v, f"{prefix}{k}."))
            elif isinstance(obj, list):
                for i, item in enumerate(obj[:20]):  # Limit array preview
                    result.extend(flatten_json(item, f"{prefix}[{i}]."))
            else:
                result.append(f"{prefix.rstrip('.')}: {str(obj)}")
            return result
        
        lines = flatten_json(data)
        return f"JSON File: {path.name}\n\n" + "\n".join(lines[:500])  # Limit output
    
    @staticmethod
    def _parse_text(path: Path) -> str:
        """Parse plain text or markdown file."""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"File: {path.name}\n\n{content}"
    
    @staticmethod
    def _parse_excel(path: Path) -> str:
        """Parse Excel file and convert to structured text."""
        xls = pd.ExcelFile(path)
        
        summaries = [f"Excel File: {path.name}"]
        summaries.append(f"Sheets: {', '.join(xls.sheet_names)}")
        summaries.append("")
        
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet_name)
            summaries.append(f"=== Sheet: {sheet_name} ===")
            summaries.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            
            # Numeric summaries
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                summaries.append(df[numeric_cols].describe().to_string())
            
            summaries.append("")
        
        return "\n".join(summaries)


class TextProcessor:
    """
    Raw text processor with preprocessing capabilities.
    """
    
    @staticmethod
    def preprocess(text: str, clean_urls: bool = True, normalize_whitespace: bool = True) -> str:
        """
        Preprocess raw text for analysis.
        
        Args:
            text: Input text
            clean_urls: Remove URLs from text
            normalize_whitespace: Normalize whitespace
            
        Returns:
            Preprocessed text
        """
        import re
        
        processed = text
        
        if clean_urls:
            url_pattern = r'https?://\S+|www\.\S+'
            processed = re.sub(url_pattern, '[URL]', processed)
        
        if normalize_whitespace:
            processed = re.sub(r'\s+', ' ', processed)
            processed = processed.strip()
        
        # Remove excessive newlines
        processed = re.sub(r'\n\s*\n', '\n\n', processed)
        
        return processed


class DataIngestionEngine:
    """
    Unified data ingestion engine supporting multiple input types.
    """
    
    def __init__(self):
        self.url_fetcher = URLFetcher()
        self.file_parser = FileParser()
        self.text_processor = TextProcessor()
    
    async def ingest(
        self,
        url: Optional[str] = None,
        file_path: Optional[str] = None,
        raw_text: Optional[str] = None
    ) -> str:
        """
        Ingest data from any supported source.
        
        Args:
            url: URL to fetch
            file_path: File path to parse
            raw_text: Raw text to process
            
        Returns:
            Processed text content
            
        Raises:
            DataIngestionError: If no valid input provided or ingestion fails
        """
        input_count = sum([
            url is not None,
            file_path is not None,
            raw_text is not None and raw_text.strip() != ""
        ])
        
        if input_count == 0:
            raise DataIngestionError("No input provided. Specify url, file_path, or raw_text.")
        
        if input_count > 1:
            raise DataIngestionError("Multiple inputs provided. Please specify only one source.")
        
        if url:
            return await self.url_fetcher.fetch(url)
        
        if file_path:
            return self.file_parser.parse(file_path)
        
        if raw_text:
            return self.text_processor.preprocess(raw_text)
        
        raise DataIngestionError("Invalid input configuration")
    
    def ingest_sync(
        self,
        url: Optional[str] = None,
        file_path: Optional[str] = None,
        raw_text: Optional[str] = None
    ) -> str:
        """Synchronous wrapper for ingest method."""
        return asyncio.run(self.ingest(url, file_path, raw_text))


# Convenience functions
async def fetch_url(url: str) -> str:
    """Fetch content from URL."""
    fetcher = URLFetcher()
    return await fetcher.fetch(url)


def parse_file(file_path: str) -> str:
    """Parse file content."""
    return FileParser.parse(file_path)


def preprocess_text(text: str) -> str:
    """Preprocess raw text."""
    return TextProcessor.preprocess(text)
