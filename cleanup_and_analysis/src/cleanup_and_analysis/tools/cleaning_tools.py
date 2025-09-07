from crewai.tools import BaseTool
from typing import Type, Dict, List, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os

class CleaningToolInput(BaseModel):
    """Input schema for CleaningTool."""
    file_path: str = Field(..., description="Path to the CSV file to clean")
    remove_duplicates: bool = Field(default=True, description="Whether to remove duplicate rows")
    handle_missing: str = Field(default="auto", description="Strategy for handling missing data: 'auto', 'drop', 'fill', 'interpolate'")
    outlier_method: str = Field(default="iqr", description="Method for outlier detection: 'iqr', 'zscore', 'none'")

class CleaningTool(BaseTool):
    name: str = "cleaning_tool"
    description: str = "Cleans CSV data by removing duplicates, handling missing values, and detecting outliers"
    args_schema: Type[BaseModel] = CleaningToolInput

    def _run(self, file_path: str, remove_duplicates: bool = True, 
             handle_missing: str = "auto", outlier_method: str = "iqr") -> str:
        """Clean CSV data and return cleaning report."""
        try:
            resolved_path = self._resolve_latest_source(file_path)
            df = pd.read_csv(resolved_path)
            original_shape = df.shape
            cleaning_report = {
                "original_shape": original_shape,
                "duplicates_removed": 0,
                "missing_values_handled": 0,
                "outliers_detected": 0,
                "final_shape": None
            }
            
            # Remove duplicates
            if remove_duplicates:
                initial_rows = len(df)
                df = df.drop_duplicates()
                cleaning_report["duplicates_removed"] = initial_rows - len(df)
            
            # Handle missing values
            missing_before = df.isnull().sum().sum()
            df = self._handle_missing_values(df, handle_missing)
            missing_after = df.isnull().sum().sum()
            cleaning_report["missing_values_handled"] = missing_before - missing_after
            
            # Detect and handle outliers
            if outlier_method != "none":
                outliers = self._detect_outliers(df, outlier_method)
                cleaning_report["outliers_detected"] = len(outliers)
                # Optionally remove outliers (commented out for now)
                # df = df.drop(outliers.index)
            
            cleaning_report["final_shape"] = df.shape
            
            # Save cleaned data
            file_name = resolved_path.split('\\')[-1].split('/')[-1]
            output_path = f"intermediate_csv/{file_name.replace('.csv', '_cleaned.csv')}"
            df.to_csv(output_path, index=False)
            
            return f"Data cleaning completed:\n" \
                   f"Original shape: {cleaning_report['original_shape']}\n" \
                   f"Duplicates removed: {cleaning_report['duplicates_removed']}\n" \
                   f"Missing values handled: {cleaning_report['missing_values_handled']}\n" \
                   f"Outliers detected: {cleaning_report['outliers_detected']}\n" \
                   f"Final shape: {cleaning_report['final_shape']}\n" \
                   f"Cleaned data saved to: {output_path}"
                   
        except Exception as e:
            return f"Error cleaning CSV file: {str(e)}"
    
    def _resolve_latest_source(self, file_path: str) -> str:
        """Prefer standardized version if available; otherwise original path."""
        base_name = os.path.basename(file_path).replace('.csv', '')
        candidates = [
            os.path.join('intermediate_csv', f"{base_name}_standardized.csv"),
            file_path
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return file_path
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values based on strategy."""
        if strategy == "auto":
            # Auto strategy: fill numeric with median, categorical with mode
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    mode_value = df[col].mode()
                    if len(mode_value) > 0:
                        df[col].fillna(mode_value[0], inplace=True)
        elif strategy == "drop":
            df = df.dropna()
        elif strategy == "fill":
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif strategy == "interpolate":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate()
        
        return df
    
    def _detect_outliers(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """Detect outliers using specified method."""
        outliers = pd.DataFrame()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = df[z_scores > 3]
            
            outliers = pd.concat([outliers, col_outliers])
        
        return outliers.drop_duplicates()

class DataQualityToolInput(BaseModel):
    """Input schema for DataQualityTool."""
    file_path: str = Field(..., description="Path to the CSV file to analyze")

class DataQualityTool(BaseTool):
    name: str = "data_quality_tool"
    description: str = "Analyzes data quality and provides detailed quality metrics"
    args_schema: Type[BaseModel] = DataQualityToolInput

    def _run(self, file_path: str) -> str:
        """Analyze data quality and return quality report."""
        try:
            df = pd.read_csv(file_path)
            
            quality_report = {
                "basic_info": {
                    "shape": df.shape,
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "columns": list(df.columns)
                },
                "completeness": {
                    "total_cells": df.size,
                    "missing_cells": df.isnull().sum().sum(),
                    "completeness_percentage": (1 - df.isnull().sum().sum() / df.size) * 100
                },
                "uniqueness": {
                    "duplicate_rows": df.duplicated().sum(),
                    "unique_rows": len(df) - df.duplicated().sum()
                },
                "consistency": {},
                "validity": {}
            }
            
            # Analyze each column
            for col in df.columns:
                col_analysis = {
                    "data_type": str(df[col].dtype),
                    "missing_count": df[col].isnull().sum(),
                    "missing_percentage": (df[col].isnull().sum() / len(df)) * 100,
                    "unique_values": df[col].nunique(),
                    "duplicate_values": len(df) - df[col].nunique()
                }
                
                if df[col].dtype in ['int64', 'float64']:
                    col_analysis.update({
                        "mean": df[col].mean(),
                        "median": df[col].median(),
                        "std": df[col].std(),
                        "min": df[col].min(),
                        "max": df[col].max()
                    })
                else:
                    col_analysis.update({
                        "most_common": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                        "most_common_count": df[col].value_counts().iloc[0] if not df[col].empty else 0
                    })
                
                quality_report["consistency"][col] = col_analysis
            
            return f"Data quality analysis completed:\n" \
                   f"Basic Info: {quality_report['basic_info']}\n" \
                   f"Completeness: {quality_report['completeness']}\n" \
                   f"Uniqueness: {quality_report['uniqueness']}\n" \
                   f"Column Analysis: {quality_report['consistency']}"
                   
        except Exception as e:
            return f"Error analyzing data quality: {str(e)}"
