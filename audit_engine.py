"""
Main audit engine orchestrating the complete analysis pipeline.
Coordinates data ingestion, multi-agent analysis, and report generation.
"""

import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from config import get_settings
from models import AnalysisRequest, ResearchReport, AnalysisMode
from utils.data_ingestion import DataIngestionEngine, DataIngestionError
from agents.qualitative_agent import QualitativeAgent
from agents.quantitative_agent import QuantitativeAgent
from agents.synthesis_agent import SynthesisAgent


class AuditEngine:
    """
    Main orchestration engine for research analysis.
    Coordinates the complete pipeline from data ingestion to report generation.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.ingestion_engine = DataIngestionEngine()
        self.qualitative_agent = QualitativeAgent()
        self.quantitative_agent = QuantitativeAgent()
        self.synthesis_agent = SynthesisAgent()
    
    def run(
        self,
        url: Optional[str] = None,
        file_path: Optional[str] = None,
        raw_text: Optional[str] = None,
        mode: AnalysisMode = AnalysisMode.MIXED,
        custom_instructions: Optional[str] = None,
        output_name: str = "analysis_report"
    ) -> ResearchReport:
        """
        Execute complete analysis pipeline.
        
        Args:
            url: URL to analyze
            file_path: File path to analyze
            raw_text: Raw text to analyze
            mode: Analysis mode (qualitative, quantitative, mixed)
            custom_instructions: Optional custom instructions
            output_name: Name for output report
            
        Returns:
            Complete ResearchReport object
            
        Raises:
            Exception: If any pipeline step fails
        """
        start_time = time.time()
        tokens_used = 0
        
        try:
            # Step 1: Data Ingestion
            print(f"[1/4] Ingesting data...")
            content = self._ingest_data(url, file_path, raw_text)
            print(f"      Ingested {len(content)} characters")
            
            # Step 2: Qualitative Analysis (if requested)
            qualitative_result = None
            if mode in [AnalysisMode.QUALITATIVE, AnalysisMode.MIXED]:
                print(f"[2/4] Running qualitative analysis...")
                try:
                    qualitative_result = self.qualitative_agent.analyze(
                        content, custom_instructions
                    )
                    print(f"      Found {len(qualitative_result.themes)} themes, "
                          f"{len(qualitative_result.entities)} entities")
                    tokens_used += self._estimate_tokens(content)
                except Exception as e:
                    print(f"      Warning: Qualitative analysis failed: {str(e)}")
            
            # Step 3: Quantitative Analysis (if requested and data available)
            quantitative_result = None
            if mode in [AnalysisMode.QUANTITATIVE, AnalysisMode.MIXED]:
                print(f"[3/4] Running quantitative analysis...")
                try:
                    # Try to extract numeric data from content
                    numeric_data = self._extract_numeric_data(content)
                    if numeric_data:
                        quantitative_result = self.quantitative_agent.analyze(
                            numeric_data, custom_instructions
                        )
                        print(f"      Analyzed {len(quantitative_result.summaries)} variables")
                        tokens_used += sum(len(v) for v in numeric_data.values()) // 4
                    else:
                        print(f"      No numeric data found for quantitative analysis")
                except Exception as e:
                    print(f"      Warning: Quantitative analysis failed: {str(e)}")
            
            # Step 4: Synthesis
            print(f"[4/4] Synthesizing findings...")
            report = self.synthesis_agent.synthesize(
                qualitative=qualitative_result,
                quantitative=quantitative_result,
                source_url=url,
                source_file=file_path,
                custom_instructions=custom_instructions
            )
            
            # Update metrics
            end_time = time.time()
            report.processing_time_ms = int((end_time - start_time) * 1000)
            report.tokens_used = tokens_used
            
            print(f"\n✓ Analysis complete in {report.processing_time_ms}ms")
            
            return report
            
        except Exception as e:
            raise Exception(f"Pipeline failed: {str(e)}")
    
    def _ingest_data(
        self,
        url: Optional[str],
        file_path: Optional[str],
        raw_text: Optional[str]
    ) -> str:
        """Ingest data from specified source."""
        if url:
            return self.ingestion_engine.ingest_sync(url=url)
        elif file_path:
            return self.ingestion_engine.ingest_sync(file_path=file_path)
        elif raw_text:
            return self.ingestion_engine.ingest_sync(raw_text=raw_text)
        else:
            raise DataIngestionError("No input source specified")
    
    def _extract_numeric_data(self, text: str) -> Dict[str, list]:
        """Extract numeric data from text for quantitative analysis."""
        import re
        
        # Look for patterns like "variable: value" or tabular data
        numeric_data = {}
        
        # Pattern 1: Key-value pairs with numbers
        kv_pattern = r'([\w\s]+):\s*(-?\d+\.?\d*)'
        matches = re.findall(kv_pattern, text)
        
        for key, value in matches:
            key = key.strip().replace(' ', '_').lower()
            if key not in numeric_data:
                numeric_data[key] = []
            try:
                numeric_data[key].append(float(value))
            except ValueError:
                pass
        
        # Pattern 2: Tab-separated or comma-separated numbers (rows of data)
        lines = text.split('\n')
        for line in lines:
            # Check for tabular data
            if '\t' in line:
                parts = line.split('\t')
                for i, part in enumerate(parts):
                    try:
                        val = float(part.strip())
                        key = f"column_{i}"
                        if key not in numeric_data:
                            numeric_data[key] = []
                        numeric_data[key].append(val)
                    except ValueError:
                        pass
        
        # Filter out columns with too few values
        numeric_data = {k: v for k, v in numeric_data.items() if len(v) >= 3}
        
        return numeric_data
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
    
    def save_report(self, report: ResearchReport, output_name: str, output_dir: str = "reports") -> str:
        """
        Save report to JSON file.
        
        Args:
            report: ResearchReport object
            output_name: Base name for output file
            output_dir: Output directory
            
        Returns:
            Path to saved file
        """
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_name}_{timestamp}.json"
        filepath = output_path / filename
        
        # Convert to dict and save
        report_dict = report.model_dump(mode='json')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📄 Report saved to: {filepath}")
        return str(filepath)


def run_analysis(request: AnalysisRequest) -> ResearchReport:
    """
    Convenience function to run analysis from request object.
    
    Args:
        request: AnalysisRequest object
        
    Returns:
        ResearchReport object
    """
    engine = AuditEngine()
    
    return engine.run(
        url=request.url,
        file_path=request.file_path,
        raw_text=request.raw_text,
        mode=request.mode,
        custom_instructions=request.custom_instructions,
        output_name=request.output_name
    )
