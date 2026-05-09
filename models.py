"""
Pydantic data models for research analysis with strict schema validation.
Defines comprehensive output structures for qualitative, quantitative, and synthesized results.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class AnalysisMode(str, Enum):
    """Analysis mode enumeration."""
    QUALITATIVE = "qualitative"
    QUANTITATIVE = "quantitative"
    MIXED = "mixed"
    SYNTHESIS = "synthesis"


class ConfidenceLevel(str, Enum):
    """Confidence level classification."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PriorityLevel(str, Enum):
    """Priority level for findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ============================================================================
# Qualitative Analysis Models
# ============================================================================

class Theme(BaseModel):
    """Represents a discovered theme in qualitative analysis."""
    name: str = Field(..., description="Theme name")
    description: str = Field(..., description="Detailed theme description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    frequency: int = Field(..., ge=0, description="Occurrence count")
    supporting_quotes: List[str] = Field(default_factory=list, description="Supporting evidence")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(..., description="Overall sentiment")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        return round(v, 3)


class Entity(BaseModel):
    """Extracted entity from text analysis."""
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity type/label")
    start_char: int = Field(..., ge=0, description="Start character position")
    end_char: int = Field(..., ge=0, description="End character position")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")


class SentimentAnalysis(BaseModel):
    """Sentiment analysis results."""
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(..., description="Overall sentiment")
    polarity_score: float = Field(..., ge=-1.0, le=1.0, description="Polarity from -1 to 1")
    subjectivity_score: float = Field(..., ge=0.0, le=1.0, description="Subjectivity from 0 to 1")
    emotion_distribution: Dict[str, float] = Field(default_factory=dict, description="Emotion scores")


class QualitativeResult(BaseModel):
    """Complete qualitative analysis result."""
    themes: List[Theme] = Field(default_factory=list, description="Discovered themes")
    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    sentiment: Optional[SentimentAnalysis] = None
    key_insights: List[str] = Field(default_factory=list, description="Key insights")
    patterns: List[str] = Field(default_factory=list, description="Identified patterns")
    anomalies: List[str] = Field(default_factory=list, description="Detected anomalies")
    word_count: int = Field(..., ge=0, description="Total word count")
    unique_concepts: int = Field(..., ge=0, description="Number of unique concepts")


# ============================================================================
# Quantitative Analysis Models
# ============================================================================

class StatisticalSummary(BaseModel):
    """Statistical summary for numeric data."""
    count: int = Field(..., ge=0, description="Sample count")
    mean: float = Field(..., description="Arithmetic mean")
    median: float = Field(..., description="Median value")
    std_dev: float = Field(..., ge=0, description="Standard deviation")
    min_value: float = Field(..., description="Minimum value")
    max_value: float = Field(..., description="Maximum value")
    q1: float = Field(..., description="First quartile")
    q3: float = Field(..., description="Third quartile")
    skewness: float = Field(..., description="Skewness coefficient")
    kurtosis: float = Field(..., description="Kurtosis coefficient")


class CorrelationMatrix(BaseModel):
    """Correlation matrix between variables."""
    variables: List[str] = Field(..., description="Variable names")
    matrix: List[List[float]] = Field(..., description="Correlation coefficients")
    
    @field_validator('matrix')
    @classmethod
    def validate_matrix(cls, v: List[List[float]], info) -> List[List[float]]:
        if not v:
            return v
        n = len(v)
        for row in v:
            if len(row) != n:
                raise ValueError("Correlation matrix must be square")
            for val in row:
                if val < -1.0 or val > 1.0:
                    raise ValueError("Correlation values must be between -1 and 1")
        return v


class DistributionAnalysis(BaseModel):
    """Distribution analysis results."""
    distribution_type: str = Field(..., description="Best-fit distribution")
    parameters: Dict[str, float] = Field(default_factory=dict, description="Distribution parameters")
    goodness_of_fit: float = Field(..., ge=0.0, le=1.0, description="Goodness of fit score")
    histogram_bins: List[float] = Field(default_factory=list, description="Histogram bin edges")
    histogram_counts: List[int] = Field(default_factory=list, description="Histogram counts")


class QuantitativeResult(BaseModel):
    """Complete quantitative analysis result."""
    summaries: Dict[str, StatisticalSummary] = Field(default_factory=dict, description="Variable summaries")
    correlations: Optional[CorrelationMatrix] = None
    distributions: Dict[str, DistributionAnalysis] = Field(default_factory=dict, description="Distribution analyses")
    outliers: List[Dict[str, Any]] = Field(default_factory=list, description="Detected outliers")
    trends: List[str] = Field(default_factory=list, description="Identified trends")
    statistical_tests: List[Dict[str, Any]] = Field(default_factory=list, description="Statistical test results")
    sample_size: int = Field(..., ge=0, description="Total sample size")
    missing_data_ratio: float = Field(..., ge=0.0, le=1.0, description="Ratio of missing data")


# ============================================================================
# Synthesis & Combined Models
# ============================================================================

class CrossValidation(BaseModel):
    """Cross-validation between qualitative and quantitative findings."""
    qualitative_theme: str = Field(..., description="Related qualitative theme")
    quantitative_support: str = Field(..., description="Supporting quantitative evidence")
    alignment_score: float = Field(..., ge=0.0, le=1.0, description="Alignment strength")
    contradictions: List[str] = Field(default_factory=list, description="Any contradictions found")


class Recommendation(BaseModel):
    """Actionable recommendation from analysis."""
    id: str = Field(..., description="Unique recommendation ID")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    priority: PriorityLevel = Field(..., description="Priority level")
    confidence: ConfidenceLevel = Field(..., description="Confidence level")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    implementation_steps: List[str] = Field(default_factory=list, description="Implementation steps")
    expected_impact: str = Field(..., description="Expected impact")


class ResearchReport(BaseModel):
    """Complete research analysis report."""
    # Metadata
    report_id: str = Field(..., description="Unique report identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")
    source_url: Optional[str] = Field(None, description="Source URL if applicable")
    source_file: Optional[str] = Field(None, description="Source file if applicable")
    analysis_mode: AnalysisMode = Field(..., description="Analysis mode used")
    
    # Analysis Results
    qualitative: Optional[QualitativeResult] = None
    quantitative: Optional[QuantitativeResult] = None
    
    # Synthesis
    cross_validations: List[CrossValidation] = Field(default_factory=list, description="Cross-validations")
    recommendations: List[Recommendation] = Field(default_factory=list, description="Recommendations")
    
    # Summary
    executive_summary: str = Field(..., description="Executive summary")
    key_findings: List[str] = Field(default_factory=list, description="Key findings")
    limitations: List[str] = Field(default_factory=list, description="Analysis limitations")
    
    # Metrics
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    tokens_used: int = Field(..., ge=0, description="Tokens consumed")
    model_used: str = Field(..., description="LLM model used")
    confidence_overall: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    
    class Config:
        json_schema_extra = {
            "example": {
                "report_id": "rpt_123456",
                "timestamp": "2024-01-15T10:30:00Z",
                "analysis_mode": "mixed",
                "executive_summary": "Comprehensive analysis reveals...",
                "key_findings": ["Finding 1", "Finding 2"],
                "processing_time_ms": 2500,
                "tokens_used": 1500,
                "model_used": "gpt-4-turbo-preview",
                "confidence_overall": 0.87
            }
        }


# ============================================================================
# Input Models
# ============================================================================

class AnalysisRequest(BaseModel):
    """Input request for analysis."""
    url: Optional[str] = Field(None, description="URL to analyze")
    file_path: Optional[str] = Field(None, description="File path to analyze")
    raw_text: Optional[str] = Field(None, description="Raw text to analyze")
    mode: AnalysisMode = Field(AnalysisMode.MIXED, description="Analysis mode")
    custom_instructions: Optional[str] = Field(None, description="Custom analysis instructions")
    output_name: str = Field(..., description="Output report name")
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v
    
    def has_input(self) -> bool:
        """Check if request has valid input."""
        return any([self.url, self.file_path, self.raw_text])
