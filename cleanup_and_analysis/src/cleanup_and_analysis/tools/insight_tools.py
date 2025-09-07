from crewai.tools import BaseTool
from typing import Type, Dict, List, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

class InsightToolInput(BaseModel):
    """Input schema for InsightTool."""
    file_path: str = Field(..., description="Path to the CSV file to analyze")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis: 'comprehensive', 'trends', 'correlations', 'patterns'")
    target_column: Optional[str] = Field(default=None, description="Target column for focused analysis")

class InsightTool(BaseTool):
    name: str = "insight_tool"
    description: str = "Analyzes data to produce insights, trends, gaps, and imbalances"
    args_schema: Type[BaseModel] = InsightToolInput

    def _run(self, file_path: str, analysis_type: str = "comprehensive", 
             target_column: Optional[str] = None) -> str:
        """Analyze data and return insights."""
        try:
            # Prefer transformed/cleaned/standardized
            resolved_path = self._resolve_latest_source(file_path)
            df = pd.read_csv(resolved_path)
            insights = {
                "data_overview": {},
                "trends": {},
                "correlations": {},
                "patterns": {},
                "gaps": {},
                "imbalances": {},
                "recommendations": []
            }
            
            # Basic data overview
            insights["data_overview"] = self._get_data_overview(df)
            
            # Trend analysis
            if analysis_type in ["comprehensive", "trends"]:
                insights["trends"] = self._analyze_trends(df, target_column)
            
            # Correlation analysis
            if analysis_type in ["comprehensive", "correlations"]:
                insights["correlations"] = self._analyze_correlations(df)
            
            # Pattern analysis
            if analysis_type in ["comprehensive", "patterns"]:
                insights["patterns"] = self._analyze_patterns(df, target_column)
            
            # Gap analysis
            insights["gaps"] = self._analyze_gaps(df)
            
            # Imbalance analysis
            insights["imbalances"] = self._analyze_imbalances(df)
            
            # Generate recommendations
            insights["recommendations"] = self._generate_recommendations(insights)
            
            # Return Markdown report content (crew will write to output_file)
            return self._format_insights(insights)
                   
        except Exception as e:
            return f"Error analyzing insights: {str(e)}"
    
    def _get_data_overview(self, df: pd.DataFrame) -> Dict:
        """Get basic data overview."""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum()
        }
    
    def _analyze_trends(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict:
        """Analyze trends in the data."""
        trends = {}
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 1:
                # Calculate trend direction
                x = np.arange(len(df[col].dropna()))
                y = df[col].dropna().values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                trends[col] = {
                    "slope": slope,
                    "r_squared": r_value ** 2,
                    "p_value": p_value,
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                    "trend_strength": "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.3 else "weak"
                }
        
        return trends
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze correlations between variables."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {"message": "Insufficient numeric columns for correlation analysis"}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Unstack the matrix to get a series of correlations
        correlations = corr_matrix.unstack().sort_values(ascending=False)
        
        # Remove self-correlations and duplicates
        correlations = correlations[correlations.index.get_level_values(0) != correlations.index.get_level_values(1)]
        
        # Get top 5 positive and negative correlations
        top_positive_correlations = correlations[correlations > 0].head(5)
        top_negative_correlations = correlations[correlations < 0].tail(5)
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "top_positive_correlations": top_positive_correlations.to_dict(),
            "top_negative_correlations": top_negative_correlations.to_dict(),
            "average_correlation": corr_matrix.abs().mean().mean()
        }
    
    def _analyze_patterns(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict:
        """Analyze patterns in the data."""
        patterns = {}
        
        # Analyze categorical patterns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() < 20:  # Likely categorical
                value_counts = df[col].value_counts()
                top_5_categories = value_counts.head(5)
                top_5_percentage = (top_5_categories.sum() / len(df)) * 100
                patterns[col] = {
                    "unique_values": df[col].nunique(),
                    "most_common": value_counts.index[0] if not value_counts.empty else None,
                    "most_common_frequency": value_counts.iloc[0] if not value_counts.empty else 0,
                    "distribution": value_counts.to_dict(),
                    "top_5_categories": top_5_categories.to_dict(),
                    "top_5_percentage": top_5_percentage
                }
        
        # Analyze temporal patterns if date columns exist
        date_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in date_cols:
            patterns[f"{col}_temporal"] = {
                "date_range": [df[col].min(), df[col].max()],
                "total_days": (df[col].max() - df[col].min()).days,
                "unique_dates": df[col].nunique()
            }
        
        return patterns
    
    def _analyze_gaps(self, df: pd.DataFrame) -> Dict:
        """Analyze gaps in the data."""
        gaps = {
            "missing_data_gaps": {},
            "temporal_gaps": {},
            "value_gaps": {}
        }
        
        # Missing data gaps
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                gaps["missing_data_gaps"][col] = {
                    "missing_count": missing_count,
                    "missing_percentage": (missing_count / len(df)) * 100
                }
        
        # Temporal gaps (if date columns exist)
        date_cols = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in date_cols:
            sorted_dates = df[col].dropna().sort_values()
            if len(sorted_dates) > 1:
                date_diffs = sorted_dates.diff().dropna()
                gaps["temporal_gaps"][col] = {
                    "average_gap": date_diffs.mean(),
                    "max_gap": date_diffs.max(),
                    "min_gap": date_diffs.min()
                }
        
        return gaps
    
    def _analyze_imbalances(self, df: pd.DataFrame) -> Dict:
        """Analyze imbalances in the data."""
        imbalances = {}
        
        # Class imbalance for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].nunique() < 20:  # Likely categorical
                value_counts = df[col].value_counts()
                if len(value_counts) > 1:
                    max_freq = value_counts.iloc[0]
                    min_freq = value_counts.iloc[-1]
                    imbalance_ratio = max_freq / min_freq
                    
                    imbalances[col] = {
                        "imbalance_ratio": imbalance_ratio,
                        "is_imbalanced": imbalance_ratio > 10,
                        "dominant_class": value_counts.index[0],
                        "minority_class": value_counts.index[-1]
                    }
        
        # Numeric distribution imbalances
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                skewness = stats.skew(df[col].dropna())
                imbalances[f"{col}_skewness"] = {
                    "skewness": skewness,
                    "is_skewed": abs(skewness) > 1,
                    "skew_direction": "right" if skewness > 0 else "left" if skewness < 0 else "symmetric"
                }
        
        return imbalances
    
    def _generate_recommendations(self, insights: Dict) -> List[str]:
        """Generate recommendations based on insights."""
        recommendations = []
        
        # Data quality recommendations
        if insights["data_overview"]["missing_values"]:
            recommendations.append("Address missing values in the dataset")
        
        if insights["data_overview"]["duplicate_rows"] > 0:
            recommendations.append("Remove duplicate rows to improve data quality")
        
        # Imbalance recommendations
        for col, imbalance in insights["imbalances"].items():
            if isinstance(imbalance, dict) and imbalance.get("is_imbalanced"):
                recommendations.append(f"Address class imbalance in {col}")
        
        # Correlation recommendations
        if insights["correlations"].get("top_positive_correlations") or \
           insights["correlations"].get("top_negative_correlations"):
            recommendations.append("Investigate strong correlations between variables")
        
        # Trend recommendations
        for col, trend in insights["trends"].items():
            if trend.get("trend_strength") == "strong":
                recommendations.append(f"Investigate strong trend in {col}")
        
        return recommendations
    
    def _format_insights(self, insights: Dict) -> str:
        """Format insights into a Markdown report."""
        report = "# Data Insights Report\n\n"
        
        # Data Overview
        report += "## Data Overview\n"
        for key, value in insights["data_overview"].items():
            report += f"- **{key}**: {value}\n"
        report += "\n"
        
        # Trends
        if insights["trends"]:
            report += "## Trend Analysis\n"
            for col, trend in insights["trends"].items():
                report += f"- **{col}**: {trend['trend_direction']} ({trend['trend_strength']})\n"
            report += "\n"
        
        # Categorical Patterns
        if insights["patterns"]:
            report += "## Categorical Patterns\n"
            for col, pattern in insights["patterns"].items():
                if "temporal" not in col:
                    report += f"- **{col}**: {pattern['unique_values']} uniques; top 5 cover {pattern['top_5_percentage']:.2f}%\n"
            report += "\n"
        
        # Correlations
        if insights["correlations"].get("top_positive_correlations"):
            report += "## Top Correlations\n"
            report += "- Top 5 Positive:\n"
            for (var1, var2), corr in insights["correlations"]["top_positive_correlations"].items():
                report += f"  - {var1} ~ {var2}: {corr:.3f}\n"
            if insights["correlations"].get("top_negative_correlations"):
                report += "- Top 5 Negative:\n"
                for (var1, var2), corr in insights["correlations"]["top_negative_correlations"].items():
                    report += f"  - {var1} ~ {var2}: {corr:.3f}\n"
            report += "\n"
        
        # Recommendations
        if insights["recommendations"]:
            report += "## Recommendations\n"
            for i, rec in enumerate(insights["recommendations"], 1):
                report += f"{i}. {rec}\n"
        
        return report

    def _resolve_latest_source(self, file_path: str) -> str:
        base_name = os.path.basename(file_path).replace('.csv', '')
        candidates = [
            os.path.join('input', f"{base_name}_transformed.csv"),
            os.path.join('intermediate_csv', f"{base_name}_engineered.csv"),
            os.path.join('intermediate_csv', f"{base_name}_cleaned.csv"),
            os.path.join('intermediate_csv', f"{base_name}_standardized.csv"),
            file_path
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return file_path
