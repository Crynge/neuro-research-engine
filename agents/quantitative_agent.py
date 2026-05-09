"""
Quantitative Analysis Agent using statistical methods and OpenAI LLM.
Performs statistical summaries, correlation analysis, distribution fitting, and trend detection.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional
from scipy import stats
from openai import OpenAI

from config import get_settings
from models import (
    QuantitativeResult, StatisticalSummary, CorrelationMatrix,
    DistributionAnalysis, AnalysisMode
)


class QuantitativeAgent:
    """
    AI-powered quantitative analysis agent.
    Combines traditional statistical methods with LLM-based interpretation.
    """
    
    INTERPRETATION_PROMPT = """You are an expert quantitative analyst and statistician.
Analyze the provided statistical summaries and data characteristics.

Your task is to:
1. Identify key trends and patterns in the data
2. Detect potential outliers and anomalies
3. Suggest relevant statistical tests
4. Provide actionable insights based on the numbers

OUTPUT REQUIREMENTS:
1. Return ONLY valid JSON matching the schema below
2. Do not include any explanatory text outside the JSON

JSON SCHEMA:
{
    "trends": ["string - identified trends"],
    "outliers": [{"variable": "string", "value": number, "z_score": number, "description": "string"}],
    "statistical_tests": [{"test_name": "string", "purpose": "string", "recommendation": "string"}],
    "insights": ["string - key quantitative insights"]
}

Consider:
- Distribution shapes and normality
- Relationships between variables
- Statistical significance thresholds
- Practical significance vs statistical significance
- Data quality issues
"""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url
        )
    
    def analyze(self, data: Dict[str, List[float]], custom_instructions: Optional[str] = None) -> QuantitativeResult:
        """
        Perform quantitative analysis on numeric data.
        
        Args:
            data: Dictionary mapping variable names to lists of numeric values
            custom_instructions: Optional additional instructions
            
        Returns:
            Structured QuantitativeResult object
        """
        # Calculate statistical summaries
        summaries = self._calculate_summaries(data)
        
        # Calculate correlation matrix
        correlations = self._calculate_correlations(data)
        
        # Fit distributions
        distributions = self._fit_distributions(data)
        
        # Get LLM-based insights
        llm_results = self._get_llm_insights(summaries, correlations, data, custom_instructions)
        
        # Calculate missing data ratio (if applicable)
        total_values = sum(len(v) for v in data.values())
        expected_values = len(data) * max(len(v) for v in data.values()) if data else 0
        missing_ratio = 1.0 - (total_values / expected_values) if expected_values > 0 else 0.0
        
        return QuantitativeResult(
            summaries=summaries,
            correlations=correlations,
            distributions=distributions,
            outliers=llm_results.get("outliers", []),
            trends=llm_results.get("trends", []),
            statistical_tests=llm_results.get("statistical_tests", []),
            sample_size=total_values,
            missing_data_ratio=missing_ratio
        )
    
    def _calculate_summaries(self, data: Dict[str, List[float]]) -> Dict[str, StatisticalSummary]:
        """Calculate comprehensive statistical summaries for each variable."""
        summaries = {}
        
        for var_name, values in data.items():
            if not values or len(values) < 2:
                continue
            
            arr = np.array(values, dtype=float)
            
            # Basic statistics
            count = len(arr)
            mean = float(np.mean(arr))
            median = float(np.median(arr))
            std_dev = float(np.std(arr, ddof=1)) if count > 1 else 0.0
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            
            # Quartiles
            q1 = float(np.percentile(arr, 25))
            q3 = float(np.percentile(arr, 75))
            
            # Higher-order moments
            skewness = float(stats.skew(arr)) if count >= 3 else 0.0
            kurtosis = float(stats.kurtosis(arr)) if count >= 4 else 0.0
            
            summaries[var_name] = StatisticalSummary(
                count=count,
                mean=round(mean, 6),
                median=round(median, 6),
                std_dev=round(std_dev, 6),
                min_value=round(min_val, 6),
                max_value=round(max_val, 6),
                q1=round(q1, 6),
                q3=round(q3, 6),
                skewness=round(skewness, 6),
                kurtosis=round(kurtosis, 6)
            )
        
        return summaries
    
    def _calculate_correlations(self, data: Dict[str, List[float]]) -> Optional[CorrelationMatrix]:
        """Calculate correlation matrix for variables with sufficient data."""
        if len(data) < 2:
            return None
        
        # Find common length
        min_len = min((len(v) for v in data.values() if len(v) > 0), default=0)
        if min_len < 3:
            return None
        
        # Prepare aligned data
        var_names = []
        aligned_data = []
        
        for var_name, values in data.items():
            if len(values) >= min_len:
                var_names.append(var_name)
                aligned_data.append(values[:min_len])
        
        if len(var_names) < 2:
            return None
        
        # Calculate correlation matrix
        try:
            corr_matrix = np.corrcoef(aligned_data)
            
            # Handle NaN values
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
            return CorrelationMatrix(
                variables=var_names,
                matrix=[[round(float(corr_matrix[i][j]), 6) for j in range(len(var_names))] 
                        for i in range(len(var_names))]
            )
        except Exception:
            return None
    
    def _fit_distributions(self, data: Dict[str, List[float]]) -> Dict[str, DistributionAnalysis]:
        """Fit probability distributions to each variable."""
        distributions = {}
        
        for var_name, values in data.items():
            if len(values) < 10:  # Need sufficient data for fitting
                continue
            
            arr = np.array(values, dtype=float)
            
            # Test multiple distributions
            dist_results = {}
            
            # Normal distribution
            try:
                ks_stat, p_value = stats.normaltest(arr)
                dist_results["normal"] = {"p_value": p_value, "ks_stat": ks_stat}
            except Exception:
                pass
            
            # Check for uniform distribution
            try:
                ks_stat, p_value = stats.kstest(arr, 'uniform', args=(arr.min(), arr.max() - arr.min()))
                dist_results["uniform"] = {"p_value": p_value, "ks_stat": ks_stat}
            except Exception:
                pass
            
            # Determine best fit
            best_dist = "unknown"
            best_p = 0.0
            
            for dist_name, results in dist_results.items():
                if results["p_value"] > best_p:
                    best_p = results["p_value"]
                    best_dist = dist_name
            
            # Generate histogram
            hist_counts, bin_edges = np.histogram(arr, bins='auto')
            
            distributions[var_name] = DistributionAnalysis(
                distribution_type=best_dist,
                parameters={"mean": float(np.mean(arr)), "std": float(np.std(arr))},
                goodness_of_fit=min(float(best_p), 1.0),
                histogram_bins=[float(b) for b in bin_edges],
                histogram_counts=[int(c) for c in hist_counts]
            )
        
        return distributions
    
    def _get_llm_insights(
        self,
        summaries: Dict[str, StatisticalSummary],
        correlations: Optional[CorrelationMatrix],
        raw_data: Dict[str, List[float]],
        custom_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get LLM-based interpretation of statistical results."""
        
        # Build context string
        context_parts = []
        
        for var_name, summary in summaries.items():
            context_parts.append(f"""
Variable: {var_name}
- Count: {summary.count}
- Mean: {summary.mean:.4f}, Median: {summary.median:.4f}
- Std Dev: {summary.std_dev:.4f}
- Range: [{summary.min_value:.4f}, {summary.max_value:.4f}]
- Skewness: {summary.skewness:.4f}, Kurtosis: {summary.kurtosis:.4f}
""")
        
        if correlations:
            context_parts.append("\nCorrelation Matrix:")
            context_parts.append(f"Variables: {correlations.variables}")
            for i, row in enumerate(correlations.matrix):
                context_parts.append(f"  {correlations.variables[i]}: {row}")
        
        user_prompt = f"""Statistical Analysis Context:
{''.join(context_parts)}

{'Additional context: ' + custom_instructions if custom_instructions else ''}

Provide your analysis in the required JSON format."""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": self.INTERPRETATION_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Warning: LLM interpretation failed: {str(e)}")
            return {"trends": [], "outliers": [], "statistical_tests": []}
    
    def analyze_from_csv_text(self, csv_text: str) -> QuantitativeResult:
        """
        Analyze quantitative data from CSV text.
        
        Args:
            csv_text: CSV formatted text
            
        Returns:
            QuantitativeResult object
        """
        import pandas as pd
        from io import StringIO
        
        df = pd.read_csv(StringIO(csv_text))
        numeric_df = df.select_dtypes(include=[np.number])
        
        data = {col: numeric_df[col].dropna().tolist() for col in numeric_df.columns}
        return self.analyze(data)
