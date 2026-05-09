"""
Synthesis Agent for cross-validating and combining qualitative and quantitative findings.
Generates unified recommendations and executive summaries.
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI

from config import get_settings
from models import (
    ResearchReport, CrossValidation, Recommendation,
    QualitativeResult, QuantitativeResult, AnalysisMode,
    PriorityLevel, ConfidenceLevel
)


class SynthesisAgent:
    """
    AI-powered synthesis agent that integrates qualitative and quantitative findings.
    Performs cross-validation, generates recommendations, and creates executive summaries.
    """
    
    SYNTHESIS_PROMPT = """You are an expert research synthesizer with deep expertise in:
- Mixed methods research integration
- Triangulation of qualitative and quantitative evidence
- Evidence-based recommendation generation
- Executive summary writing for technical and non-technical audiences

Your task is to synthesize the provided qualitative and quantitative findings into a coherent report.

INPUT DATA:
- Qualitative findings: themes, entities, sentiments, patterns
- Quantitative findings: statistical summaries, correlations, trends

OUTPUT REQUIREMENTS:
1. Return ONLY valid JSON matching the exact schema below
2. Do not include any explanatory text outside the JSON
3. Ensure all cross-validations are evidence-based
4. Generate actionable, prioritized recommendations

JSON SCHEMA:
{
    "cross_validations": [
        {
            "qualitative_theme": "string",
            "quantitative_support": "string",
            "alignment_score": float (0-1),
            "contradictions": ["string"]
        }
    ],
    "recommendations": [
        {
            "id": "string",
            "title": "string",
            "description": "string",
            "priority": "critical|high|medium|low",
            "confidence": "high|medium|low",
            "evidence": ["string"],
            "implementation_steps": ["string"],
            "expected_impact": "string"
        }
    ],
    "executive_summary": "string (2-4 paragraphs)",
    "key_findings": ["string (5-10 key findings)"],
    "limitations": ["string"],
    "confidence_overall": float (0-1)
}

SYNTHESIS GUIDELINES:
- Look for convergence between qualitative themes and quantitative patterns
- Identify divergences that need further investigation
- Prioritize recommendations based on evidence strength and potential impact
- Write clear, actionable insights avoiding jargon
- Acknowledge limitations honestly
"""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url
        )
    
    def synthesize(
        self,
        qualitative: Optional[QualitativeResult],
        quantitative: Optional[QuantitativeResult],
        source_url: Optional[str] = None,
        source_file: Optional[str] = None,
        custom_instructions: Optional[str] = None
    ) -> ResearchReport:
        """
        Synthesize qualitative and quantitative findings into unified report.
        
        Args:
            qualitative: Qualitative analysis results
            quantitative: Quantitative analysis results
            source_url: Original source URL if applicable
            source_file: Original source file if applicable
            custom_instructions: Optional additional instructions
            
        Returns:
            Complete ResearchReport object
        """
        # Determine analysis mode
        if qualitative and quantitative:
            mode = AnalysisMode.MIXED
        elif qualitative:
            mode = AnalysisMode.QUALITATIVE
        elif quantitative:
            mode = AnalysisMode.QUANTITATIVE
        else:
            raise ValueError("At least one analysis type must be provided")
        
        # Get LLM synthesis
        synthesis_results = self._get_synthesis(qualitative, quantitative, custom_instructions)
        
        # Build cross-validations
        cross_validations = [
            CrossValidation(**cv) for cv in synthesis_results.get("cross_validations", [])
        ]
        
        # Build recommendations
        recommendations = []
        for rec in synthesis_results.get("recommendations", []):
            recommendations.append(Recommendation(
                id=rec.get("id", f"rec_{uuid.uuid4().hex[:8]}"),
                title=rec.get("title", "Untitled Recommendation"),
                description=rec.get("description", ""),
                priority=PriorityLevel(rec.get("priority", "medium")),
                confidence=ConfidenceLevel(rec.get("confidence", "medium")),
                evidence=rec.get("evidence", []),
                implementation_steps=rec.get("implementation_steps", []),
                expected_impact=rec.get("expected_impact", "")
            ))
        
        # Calculate processing metrics (placeholder - actual values set by engine)
        return ResearchReport(
            report_id=f"rpt_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.utcnow(),
            source_url=source_url,
            source_file=source_file,
            analysis_mode=mode,
            qualitative=qualitative,
            quantitative=quantitative,
            cross_validations=cross_validations,
            recommendations=recommendations,
            executive_summary=synthesis_results.get("executive_summary", ""),
            key_findings=synthesis_results.get("key_findings", []),
            limitations=synthesis_results.get("limitations", []),
            processing_time_ms=0,  # Set by engine
            tokens_used=0,  # Set by engine
            model_used=self.settings.openai_model,
            confidence_overall=float(synthesis_results.get("confidence_overall", 0.5))
        )
    
    def _get_synthesis(
        self,
        qualitative: Optional[QualitativeResult],
        quantitative: Optional[QuantitativeResult],
        custom_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get LLM-based synthesis of findings."""
        
        # Build context from qualitative results
        qual_context = ""
        if qualitative:
            qual_parts = ["=== QUALITATIVE FINDINGS ==="]
            
            if qualitative.themes:
                qual_parts.append("\nThemes:")
                for theme in qualitative.themes:
                    qual_parts.append(f"- {theme.name}: {theme.description} (confidence: {theme.confidence})")
            
            if qualitative.sentiment:
                s = qualitative.sentiment
                qual_parts.append(f"\nSentiment: {s.overall_sentiment} (polarity: {s.polarity_score}, subjectivity: {s.subjectivity_score})")
            
            if qualitative.key_insights:
                qual_parts.append("\nKey Insights:")
                for insight in qualitative.key_insights:
                    qual_parts.append(f"- {insight}")
            
            if qualitative.patterns:
                qual_parts.append("\nPatterns:")
                for pattern in qualitative.patterns:
                    qual_parts.append(f"- {pattern}")
            
            qual_context = "\n".join(qual_parts)
        
        # Build context from quantitative results
        quant_context = ""
        if quantitative:
            quant_parts = ["=== QUANTITATIVE FINDINGS ==="]
            
            if quantitative.summaries:
                quant_parts.append("\nStatistical Summaries:")
                for var, summary in quantitative.summaries.items():
                    quant_parts.append(f"- {var}: mean={summary.mean:.2f}, std={summary.std_dev:.2f}, n={summary.count}")
            
            if quantitative.correlations:
                quant_parts.append(f"\nCorrelations: {len(quantitative.correlations.variables)} variables analyzed")
            
            if quantitative.trends:
                quant_parts.append("\nTrends:")
                for trend in quantitative.trends:
                    quant_parts.append(f"- {trend}")
            
            quant_context = "\n".join(quant_parts)
        
        user_prompt = f"""{qual_context}

{quant_context}

{'Additional instructions: ' + custom_instructions if custom_instructions else ''}

Generate a comprehensive synthesis following the specified JSON schema."""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": self.SYNTHESIS_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Warning: Synthesis failed: {str(e)}")
            return self._generate_fallback_synthesis(qualitative, quantitative)
    
    def _generate_fallback_synthesis(
        self,
        qualitative: Optional[QualitativeResult],
        quantitative: Optional[QuantitativeResult]
    ) -> Dict[str, Any]:
        """Generate basic synthesis without LLM when API fails."""
        
        key_findings = []
        
        if qualitative:
            for theme in qualitative.themes[:3]:
                key_findings.append(f"Theme identified: {theme.name}")
            if qualitative.sentiment:
                key_findings.append(f"Overall sentiment: {qualitative.sentiment.overall_sentiment}")
        
        if quantitative:
            for var, summary in list(quantitative.summaries.items())[:3]:
                key_findings.append(f"{var}: mean={summary.mean:.2f} (n={summary.count})")
        
        return {
            "cross_validations": [],
            "recommendations": [],
            "executive_summary": "Analysis completed. See detailed findings for specifics.",
            "key_findings": key_findings,
            "limitations": ["LLM synthesis unavailable"],
            "confidence_overall": 0.5
        }
