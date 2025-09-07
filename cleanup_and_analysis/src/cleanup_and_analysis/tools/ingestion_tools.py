from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import pandas as pd
import os
from pathlib import Path

class IngestionToolInput(BaseModel):
    """Input schema for IngestionTool."""
    file_path: str = Field(..., description="Path to the CSV file to ingest")
    encoding: str = Field(default="utf-8", description="File encoding (default: utf-8)")

class IngestionTool(BaseTool):
    name: str = "csv_ingestion_tool"
    description: str = "Ingests CSV data from file path and returns basic information about the dataset"
    args_schema: Type[BaseModel] = IngestionToolInput

    def _run(self, file_path: str, encoding: str = "utf-8") -> str:
        """Ingest CSV file and return dataset information."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return f"Error: File {file_path} not found"
            
            # Read CSV file
            df = pd.read_csv(file_path, encoding=encoding)
            
            # Get basic information
            info = {
                "file_path": file_path,
                "shape": df.shape,
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "sample_data": df.head(3).to_dict()
            }
            
            return f"Successfully ingested CSV file:\n" \
                   f"Shape: {info['shape']}\n" \
                   f"Columns: {info['columns']}\n" \
                   f"Data types: {info['data_types']}\n" \
                   f"Memory usage: {info['memory_usage']} bytes\n" \
                   f"Sample data: {info['sample_data']}"
                   
        except Exception as e:
            return f"Error ingesting CSV file: {str(e)}"

class DataValidationToolInput(BaseModel):
    """Input schema for DataValidationTool."""
    file_path: str = Field(..., description="Path to the CSV file to validate")

class DataValidationTool(BaseTool):
    name: str = "data_validation_tool"
    description: str = "Validates CSV data structure and identifies potential issues"
    args_schema: Type[BaseModel] = DataValidationToolInput

    def _run(self, file_path: str) -> str:
        """Validate CSV data and return validation report."""
        try:
            df = pd.read_csv(file_path)
            
            validation_report = {
                "file_path": file_path,
                "is_valid": True,
                "issues": [],
                "warnings": []
            }
            
            # Check for empty dataset
            if df.empty:
                validation_report["is_valid"] = False
                validation_report["issues"].append("Dataset is empty")
            
            # Check for duplicate column names
            if len(df.columns) != len(set(df.columns)):
                validation_report["warnings"].append("Duplicate column names detected")
            
            # Check for completely empty columns
            empty_columns = df.columns[df.isnull().all()].tolist()
            if empty_columns:
                validation_report["warnings"].append(f"Empty columns: {empty_columns}")
            
            # Check for mixed data types in columns
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if numeric data is stored as strings
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except:
                        pass  # This is expected for non-numeric data
                    else:
                        validation_report["warnings"].append(f"Column '{col}' contains numeric data stored as strings")
            
            return f"Data validation completed:\n" \
                   f"Valid: {validation_report['is_valid']}\n" \
                   f"Issues: {validation_report['issues']}\n" \
                   f"Warnings: {validation_report['warnings']}"
                   
        except Exception as e:
            return f"Error validating CSV file: {str(e)}"
