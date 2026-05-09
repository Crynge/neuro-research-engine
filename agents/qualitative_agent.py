"""
Qualitative Analysis Agent using OpenAI LLM.
Performs thematic analysis, entity extraction, sentiment detection, and pattern recognition.
"""

import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI

from config import get_settings
from models import (
    QualitativeResult, Theme, Entity, SentimentAnalysis,
    ConfidenceLevel, AnalysisMode
)


class QualitativeAgent:
    """
    AI-powered qualitative analysis agent.
    Extracts themes, entities, sentiments, patterns, and insights from text data.
    """
    
    SYSTEM_PROMPT = """You are an expert qualitative research analyst with deep expertise in:
- Thematic analysis and coding
- Grounded theory methodology
- Sentiment and emotion analysis
- Entity recognition and relationship mapping
- Pattern detection and anomaly identification

Your task is to analyze the provided text and extract structured qualitative findings.

OUTPUT REQUIREMENTS:
1. Return ONLY valid JSON matching the exact schema below
2. Do not include any explanatory text outside the JSON
3. Ensure all required fields are present
4. Use high confidence scores only for well-supported findings

JSON SCHEMA:
{
    "themes": [
        {
            "name": "string",
            "description": "string",
            "confidence": float (0-1),
            "frequency": int,
            "supporting_quotes": ["string"],
            "sentiment": "positive|negative|neutral|mixed"
        }
    ],
    "entities": [
        {
            "text": "string",
            "label": "string (PERSON|ORGANIZATION|LOCATION|DATE|CONCEPT|etc)",
            "start_char": int,
            "end_char": int,
            "confidence": float (0-1)
        }
    ],
    "sentiment": {
        "overall_sentiment": "positive|negative|neutral|mixed",
        "polarity_score": float (-1 to 1),
        "subjectivity_score": float (0 to 1),
        "emotion_distribution": {"joy": float, "anger": float, "fear": float, "sadness": float, "surprise": float}
    },
    "key_insights": ["string"],
    "patterns": ["string"],
    "anomalies": ["string"],
    "word_count": int,
    "unique_concepts": int
}

ANALYSIS GUIDELINES:
- Identify 3-7 major themes with strong supporting evidence
- Extract entities with precise character positions
- Calculate sentiment based on linguistic markers
- Detect recurring patterns and notable anomalies
- Provide actionable insights grounded in the data
"""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url
        )
    
    def analyze(self, text: str, custom_instructions: Optional[str] = None) -> QualitativeResult:
        """
        Perform qualitative analysis on text.
        
        Args:
            text: Input text to analyze
            custom_instructions: Optional additional instructions
            
        Returns:
            Structured QualitativeResult object
            
        Raises:
            Exception: If analysis fails
        """
        user_prompt = f"""Analyze the following text using rigorous qualitative methods:

{text}

{'Additional instructions: ' + custom_instructions if custom_instructions else ''}

Remember: Return ONLY valid JSON matching the specified schema."""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.settings.openai_max_tokens,
                temperature=self.settings.openai_temperature,
                response_format={"type": "json_object"}
            )
            
            result_json = json.loads(response.choices[0].message.content)
            return self._parse_result(result_json, text)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from LLM: {str(e)}")
        except Exception as e:
            raise Exception(f"Qualitative analysis failed: {str(e)}")
    
    def _parse_result(self, data: Dict[str, Any], original_text: str) -> QualitativeResult:
        """Parse LLM response into validated QualitativeResult model."""
        
        # Parse themes
        themes = []
        for t in data.get("themes", []):
            themes.append(Theme(
                name=t.get("name", "Unknown"),
                description=t.get("description", ""),
                confidence=float(t.get("confidence", 0.5)),
                frequency=int(t.get("frequency", 1)),
                supporting_quotes=t.get("supporting_quotes", []),
                sentiment=t.get("sentiment", "neutral")
            ))
        
        # Parse entities - estimate positions if not provided
        entities = []
        for e in data.get("entities", []):
            start = e.get("start_char", 0)
            end = e.get("end_char", len(e.get("text", "")))
            
            # Validate positions against original text
            if start > len(original_text) or end > len(original_text):
                # Estimate position by searching
                entity_text = e.get("text", "")
                pos = original_text.find(entity_text)
                if pos >= 0:
                    start = pos
                    end = pos + len(entity_text)
                else:
                    start = 0
                    end = len(entity_text)
            
            entities.append(Entity(
                text=e.get("text", ""),
                label=e.get("label", "CONCEPT"),
                start_char=start,
                end_char=end,
                confidence=float(e.get("confidence", 0.5))
            ))
        
        # Parse sentiment
        sentiment_data = data.get("sentiment", {})
        sentiment = SentimentAnalysis(
            overall_sentiment=sentiment_data.get("overall_sentiment", "neutral"),
            polarity_score=float(sentiment_data.get("polarity_score", 0.0)),
            subjectivity_score=float(sentiment_data.get("subjectivity_score", 0.5)),
            emotion_distribution=sentiment_data.get("emotion_distribution", {})
        )
        
        # Create result
        return QualitativeResult(
            themes=themes,
            entities=entities,
            sentiment=sentiment,
            key_insights=data.get("key_insights", []),
            patterns=data.get("patterns", []),
            anomalies=data.get("anomalies", []),
            word_count=data.get("word_count", len(original_text.split())),
            unique_concepts=data.get("unique_concepts", len(set(original_text.split())))
        )
    
    def analyze_batch(self, texts: List[str]) -> List[QualitativeResult]:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of QualitativeResult objects
        """
        results = []
        for text in texts:
            try:
                result = self.analyze(text)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to analyze text segment: {str(e)}")
                results.append(None)
        return results
