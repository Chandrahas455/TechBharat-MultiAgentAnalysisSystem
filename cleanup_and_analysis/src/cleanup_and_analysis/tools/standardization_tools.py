from crewai.tools import BaseTool
from typing import Type, Dict, List
from pydantic import BaseModel, Field
import pandas as pd
import re

class StandardizationToolInput(BaseModel):
    """Input schema for StandardizationTool."""
    file_path: str = Field(..., description="Path to the CSV file to standardize")
    column_mapping: Dict[str, str] = Field(default={}, description="Dictionary mapping old column names to new standardized names")

class StandardizationTool(BaseTool):
    name: str = "standardization_tool"
    description: str = "Standardizes column names, formats, and data types in CSV files"
    args_schema: Type[BaseModel] = StandardizationToolInput

    def _run(self, file_path: str, column_mapping: Dict[str, str] = {}) -> str:
        """Standardize CSV data and return standardized version."""
        try:
            df = pd.read_csv(file_path)
            original_columns = df.columns.tolist()
            
            # Standardize column names
            if column_mapping:
                df = df.rename(columns=column_mapping)
            else:
                # Auto-standardize column names
                df.columns = [self._standardize_column_name(col) for col in df.columns]
            
            # Standardize data types
            df = self._standardize_data_types(df)
            
            # Save standardized data
            file_name = file_path.split('\\')[-1].split('/')[-1]
            output_path = f"intermediate_csv/{file_name.replace('.csv', '_standardized.csv')}"
            df.to_csv(output_path, index=False)
            
            return f"Standardization completed:\n" \
                   f"Original columns: {original_columns}\n" \
                   f"Standardized columns: {df.columns.tolist()}\n" \
                   f"Output saved to: {output_path}\n" \
                   f"Data types: {df.dtypes.to_dict()}"
                   
        except Exception as e:
            return f"Error standardizing CSV file: {str(e)}"
    
    def _standardize_column_name(self, name: str) -> str:
        """Standardize column name format."""
        # Convert to lowercase
        name = name.lower()
        # Replace spaces and special characters with underscores
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name
    
    def _standardize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types in the dataframe."""
        for col in df.columns:
            # Try to convert to numeric
            if df[col].dtype == 'object':
                # Check if it's a date
                if self._is_date_column(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                # Check if it's numeric
                elif self._is_numeric_column(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a column contains date data."""
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        try:
            pd.to_datetime(sample, errors='raise')
            return True
        except:
            return False
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column contains numeric data."""
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        try:
            pd.to_numeric(sample, errors='raise')
            return True
        except:
            return False

class ColumnAnalysisToolInput(BaseModel):
    """Input schema for ColumnAnalysisTool."""
    file_path: str = Field(..., description="Path to the CSV file to analyze")

class ColumnAnalysisTool(BaseTool):
    name: str = "column_analysis_tool"
    description: str = "Analyzes columns to suggest standardization improvements"
    args_schema: Type[BaseModel] = ColumnAnalysisToolInput

    def _run(self, file_path: str) -> str:
        """Analyze columns and suggest standardization improvements."""
        try:
            df = pd.read_csv(file_path)
            
            analysis = {
                "column_issues": [],
                "suggestions": []
            }
            
            for col in df.columns:
                col_analysis = {
                    "column": col,
                    "issues": [],
                    "suggestions": []
                }
                
                # Check column name format
                if not re.match(r'^[a-z][a-z0-9_]*$', col.lower()):
                    col_analysis["issues"].append("Non-standard column name format")
                    col_analysis["suggestions"].append(f"Rename to: {self._standardize_column_name(col)}")
                
                # Check data type consistency
                if df[col].dtype == 'object':
                    if self._is_numeric_column(df[col]):
                        col_analysis["issues"].append("Numeric data stored as strings")
                        col_analysis["suggestions"].append("Convert to numeric type")
                    elif self._is_date_column(df[col]):
                        col_analysis["issues"].append("Date data stored as strings")
                        col_analysis["suggestions"].append("Convert to datetime type")
                
                # Check for mixed case in categorical data
                if df[col].dtype == 'object' and not self._is_numeric_column(df[col]):
                    unique_values = df[col].dropna().unique()
                    if len(unique_values) < 20:  # Likely categorical
                        case_variations = set([str(val).lower() for val in unique_values])
                        if len(case_variations) < len(unique_values):
                            col_analysis["issues"].append("Mixed case in categorical data")
                            col_analysis["suggestions"].append("Standardize case (lowercase recommended)")
                
                if col_analysis["issues"]:
                    analysis["column_issues"].append(col_analysis)
            
            return f"Column analysis completed:\n" \
                   f"Total columns: {len(df.columns)}\n" \
                   f"Columns with issues: {len(analysis['column_issues'])}\n" \
                   f"Detailed analysis: {analysis['column_issues']}"
                   
        except Exception as e:
            return f"Error analyzing columns: {str(e)}"
    
    def _standardize_column_name(self, name: str) -> str:
        """Standardize column name format."""
        name = name.lower()
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        name = re.sub(r'_+', '_', name)
        name = name.strip('_')
        return name
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a column contains date data."""
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        try:
            pd.to_datetime(sample, errors='raise')
            return True
        except:
            return False
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a column contains numeric data."""
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        try:
            pd.to_numeric(sample, errors='raise')
            return True
        except:
            return False
