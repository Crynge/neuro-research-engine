#!/usr/bin/env python3
"""
Neuro Research Engine - CLI Entry Point

Advanced AI-powered research analysis for qualitative and quantitative data.
Supports URL, file, and raw text inputs with multi-agent analysis pipeline.

Usage:
    python main.py --url "https://example.com" --output report_name
    python main.py --file data.csv --mode quantitative --output stats
    python main.py --text "Your text here" --mode qualitative --output themes
    python main.py --url "https://example.com" --mode mixed --output full_analysis
"""

import sys
import argparse
from typing import Optional
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from config import get_settings, reload_settings
from models import AnalysisRequest, AnalysisMode, ResearchReport
from audit_engine import AuditEngine


console = Console()


def print_banner():
    """Print application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║     NEURO RESEARCH ENGINE v2.0                            ║
║     Advanced AI-Powered Research Analysis                 ║
║     Qualitative + Quantitative + Synthesis                ║
╚═══════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))


def print_summary_table(report: ResearchReport):
    """Print formatted summary table of analysis results."""
    
    # Metadata Table
    meta_table = Table(title="📊 Analysis Metadata", box=box.ROUNDED)
    meta_table.add_column("Property", style="cyan")
    meta_table.add_column("Value", style="green")
    
    meta_table.add_row("Report ID", report.report_id)
    meta_table.add_row("Timestamp", report.timestamp.strftime("%Y-%m-%d %H:%M:%S"))
    meta_table.add_row("Mode", report.analysis_mode.value)
    meta_table.add_row("Processing Time", f"{report.processing_time_ms}ms")
    meta_table.add_row("Tokens Used", str(report.tokens_used))
    meta_table.add_row("Model", report.model_used)
    meta_table.add_row("Overall Confidence", f"{report.confidence_overall:.2%}")
    
    if report.source_url:
        meta_table.add_row("Source URL", report.source_url)
    if report.source_file:
        meta_table.add_row("Source File", report.source_file)
    
    console.print(meta_table)
    console.print()
    
    # Key Findings Table
    if report.key_findings:
        findings_table = Table(title="🔍 Key Findings", box=box.ROUNDED)
        findings_table.add_column("#", style="yellow")
        findings_table.add_column("Finding", style="white")
        
        for i, finding in enumerate(report.key_findings[:10], 1):
            findings_table.add_row(str(i), finding)
        
        console.print(findings_table)
        console.print()
    
    # Qualitative Summary
    if report.qualitative:
        qual_table = Table(title="📝 Qualitative Analysis", box=box.ROUNDED)
        qual_table.add_column("Metric", style="cyan")
        qual_table.add_column("Value", style="green")
        
        qual_table.add_row("Themes Found", str(len(report.qualitative.themes)))
        qual_table.add_row("Entities Extracted", str(len(report.qualitative.entities)))
        qual_table.add_row("Word Count", str(report.qualitative.word_count))
        qual_table.add_row("Unique Concepts", str(report.qualitative.unique_concepts))
        
        if report.qualitative.sentiment:
            s = report.qualitative.sentiment
            qual_table.add_row("Sentiment", s.overall_sentiment)
            qual_table.add_row("Polarity", f"{s.polarity_score:.3f}")
            qual_table.add_row("Subjectivity", f"{s.subjectivity_score:.3f}")
        
        console.print(qual_table)
        console.print()
    
    # Quantitative Summary
    if report.quantitative:
        quant_table = Table(title="📈 Quantitative Analysis", box=box.ROUNDED)
        quant_table.add_column("Metric", style="cyan")
        quant_table.add_column("Value", style="green")
        
        quant_table.add_row("Variables Analyzed", str(len(report.quantitative.summaries)))
        quant_table.add_row("Sample Size", str(report.quantitative.sample_size))
        quant_table.add_row("Missing Data Ratio", f"{report.quantitative.missing_data_ratio:.2%}")
        
        if report.quantitative.correlations:
            n_vars = len(report.quantitative.correlations.variables)
            quant_table.add_row("Correlation Matrix", f"{n_vars}x{n_vars}")
        
        if report.quantitative.distributions:
            quant_table.add_row("Distributions Fitted", str(len(report.quantitative.distributions)))
        
        console.print(quant_table)
        console.print()
    
    # Recommendations
    if report.recommendations:
        rec_table = Table(title="💡 Recommendations", box=box.ROUNDED)
        rec_table.add_column("Priority", style="bold")
        rec_table.add_column("Title", style="cyan")
        rec_table.add_column("Confidence", style="green")
        
        priority_icons = {
            "critical": "🔴 CRITICAL",
            "high": "🟠 HIGH",
            "medium": "🟡 MEDIUM",
            "low": "🟢 LOW"
        }
        
        for rec in report.recommendations[:5]:
            icon = priority_icons.get(rec.priority.value, rec.priority.value)
            rec_table.add_row(icon, rec.title, rec.confidence.value)
        
        console.print(rec_table)
        console.print()
    
    # Executive Summary Panel
    if report.executive_summary:
        console.print(Panel(
            report.executive_summary[:2000] + ("..." if len(report.executive_summary) > 2000 else ""),
            title="📋 Executive Summary",
            style="bold green"
        ))


def print_limitations(report: ResearchReport):
    """Print limitations if any."""
    if report.limitations:
        console.print("\n[bold yellow]⚠️  Limitations:[/bold yellow]")
        for limit in report.limitations:
            console.print(f"  • {limit}")


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command-line arguments."""
    input_count = sum([
        args.url is not None,
        args.file is not None,
        args.text is not None and args.text.strip() != ""
    ])
    
    if input_count == 0:
        console.print("[bold red]Error:[/bold red] Must specify one of: --url, --file, or --text")
        return False
    
    if input_count > 1:
        console.print("[bold red]Error:[/bold red] Cannot specify multiple input sources")
        return False
    
    return True


def create_analysis_request(args: argparse.Namespace) -> AnalysisRequest:
    """Create AnalysisRequest from parsed arguments."""
    mode_map = {
        "qualitative": AnalysisMode.QUALITATIVE,
        "quantitative": AnalysisMode.QUANTITATIVE,
        "mixed": AnalysisMode.MIXED
    }
    
    return AnalysisRequest(
        url=args.url,
        file_path=args.file,
        raw_text=args.text,
        mode=mode_map.get(args.mode, AnalysisMode.MIXED),
        custom_instructions=args.instructions,
        output_name=args.output
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Neuro Research Engine - AI-Powered Research Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url "https://example.com/article" --output article_analysis
  %(prog)s --file survey_results.csv --mode quantitative --output survey
  %(prog)s --text "Customer feedback text..." --mode qualitative --output feedback
  %(prog)s --url "https://example.com" --mode mixed --instructions "Focus on sentiment" --output full
        """
    )
    
    # Input options (mutually exclusive group handled in validation)
    input_group = parser.add_argument_group("Input Options (choose one)")
    input_group.add_argument("--url", "-u", type=str, help="URL to analyze")
    input_group.add_argument("--file", "-f", type=str, help="File path to analyze (CSV, JSON, TXT, XLSX)")
    input_group.add_argument("--text", "-t", type=str, help="Raw text to analyze")
    
    # Analysis options
    analysis_group = parser.add_argument_group("Analysis Options")
    analysis_group.add_argument(
        "--mode", "-m",
        type=str,
        choices=["qualitative", "quantitative", "mixed"],
        default="mixed",
        help="Analysis mode (default: mixed)"
    )
    analysis_group.add_argument(
        "--instructions", "-i",
        type=str,
        help="Custom instructions for the analysis"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        default="analysis_report",
        help="Output report name (default: analysis_report)"
    )
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory (default: reports)"
    )
    
    # System options
    system_group = parser.add_argument_group("System Options")
    system_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    system_group.add_argument(
        "--reload-config",
        action="store_true",
        help="Reload configuration from .env file"
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Validate arguments
    if not validate_args(args):
        parser.print_help()
        sys.exit(1)
    
    # Reload config if requested
    if args.reload_config:
        console.print("[cyan]Reloading configuration...[/cyan]")
        reload_settings()
    
    # Check API key
    try:
        settings = get_settings()
        if not settings.openai_api_key or settings.openai_api_key == "sk-your-api-key-here":
            console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not configured")
            console.print("Please copy .env.example to .env and set your API key")
            sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {str(e)}")
        sys.exit(1)
    
    # Create analysis request
    request = create_analysis_request(args)
    
    # Run analysis
    try:
        console.print(f"\n[cyan]Starting analysis...[/cyan]")
        console.print(f"  Mode: [bold]{request.mode.value}[/bold]")
        
        if request.url:
            console.print(f"  Source: [bold]{request.url}[/bold]")
        elif request.file_path:
            console.print(f"  Source: [bold]{request.file_path}[/bold]")
        
        console.print()
        
        # Initialize and run engine
        engine = AuditEngine()
        report = engine.run(
            url=request.url,
            file_path=request.file_path,
            raw_text=request.raw_text,
            mode=request.mode,
            custom_instructions=request.custom_instructions,
            output_name=request.output
        )
        
        # Save report
        output_path = engine.save_report(report, request.output, args.output_dir)
        
        # Print summary
        console.print("\n")
        print_summary_table(report)
        
        # Print limitations
        print_limitations(report)
        
        console.print(f"\n[bold green]✓ Analysis completed successfully![/bold green]")
        console.print(f"[dim]Report saved to: {output_path}[/dim]")
        
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Analysis interrupted by user[/bold yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
