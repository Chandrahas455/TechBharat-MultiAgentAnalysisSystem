from crewai.tools import BaseTool
from typing import Type, Dict, List, Optional
from pydantic import BaseModel, Field
import pandas as pd
import json
from datetime import datetime

class DocumentationToolInput(BaseModel):
    """Input schema for DocumentationTool."""
    file_path: str = Field(..., description="Path to the CSV file to document")
    process_log: Dict = Field(default={}, description="Log of processing steps performed")
    output_directory: str = Field(default="./", description="Directory to save documentation")

class DocumentationTool(BaseTool):
    name: str = "documentation_tool"
    description: str = "Creates comprehensive documentation of data processing steps and changes"
    args_schema: Type[BaseModel] = DocumentationToolInput

    def _run(self, file_path: str, process_log: Dict = {}, output_directory: str = "./") -> str:
        """Create documentation and return documentation report."""
        try:
            df = pd.read_csv(file_path)
            
            # Generate documentation
            doc_report = {
                "file_info": self._get_file_info(file_path, df),
                "data_summary": self._get_data_summary(df),
                "processing_steps": process_log,
                "data_dictionary": self._create_data_dictionary(df),
                "quality_metrics": self._get_quality_metrics(df),
                "recommendations": self._get_documentation_recommendations(df)
            }
            
            # Save documentation files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON documentation
            json_path = f"intermediate_csv/data_documentation_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(doc_report, f, indent=2, default=str)
            
            # Save markdown documentation
            md_path = "report.md"
            with open(md_path, 'w') as f:
                f.write(self._create_markdown_documentation(doc_report))
            
            # Save data dictionary as CSV
            dict_path = f"intermediate_csv/data_dictionary_{timestamp}.csv"
            data_dict_df = pd.DataFrame(doc_report["data_dictionary"])
            data_dict_df.to_csv(dict_path, index=False)
            
            return f"Documentation created successfully:\n" \
                   f"File info: {doc_report['file_info']}\n" \
                   f"Data summary: {doc_report['data_summary']}\n" \
                   f"Processing steps logged: {len(process_log)}\n" \
                   f"Data dictionary entries: {len(doc_report['data_dictionary'])}\n" \
                   f"Files saved:\n" \
                   f"  - JSON: {json_path}\n" \
                   f"  - Markdown: {md_path}\n" \
                   f"  - Data Dictionary: {dict_path}"
                   
        except Exception as e:
            return f"Error creating documentation: {str(e)}"
    
    def _get_file_info(self, file_path: str, df: pd.DataFrame) -> Dict:
        """Get basic file information."""
        return {
            "file_path": file_path,
            "file_name": file_path.split('/')[-1],
            "creation_timestamp": datetime.now().isoformat(),
            "file_size_bytes": df.memory_usage(deep=True).sum(),
            "encoding": "utf-8"  # Default assumption
        }
    
    def _get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get data summary information."""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": df.memory_usage(deep=True).sum(),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "numeric_columns": list(df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(df.select_dtypes(include=['object']).columns),
            "datetime_columns": list(df.select_dtypes(include=['datetime64[ns]']).columns)
        }
    
    def _create_data_dictionary(self, df: pd.DataFrame) -> List[Dict]:
        """Create a data dictionary for all columns."""
        data_dict = []
        
        for col in df.columns:
            col_info = {
                "column_name": col,
                "data_type": str(df[col].dtype),
                "description": self._generate_column_description(col, df[col]),
                "missing_count": df[col].isnull().sum(),
                "missing_percentage": (df[col].isnull().sum() / len(df)) * 100,
                "unique_values": df[col].nunique(),
                "sample_values": df[col].dropna().head(5).tolist()
            }
            
            # Add type-specific information
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    "min_value": df[col].min(),
                    "max_value": df[col].max(),
                    "mean_value": df[col].mean(),
                    "median_value": df[col].median(),
                    "std_value": df[col].std()
                })
            elif df[col].dtype == 'object':
                col_info.update({
                    "most_common_value": df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    "most_common_frequency": df[col].value_counts().iloc[0] if not df[col].empty else 0
                })
            
            data_dict.append(col_info)
        
        return data_dict
    
    def _generate_column_description(self, col_name: str, series: pd.Series) -> str:
        """Generate a description for a column based on its name and content."""
        col_lower = col_name.lower()
        
        # Common patterns for column descriptions
        if any(word in col_lower for word in ['id', 'key', 'index']):
            return f"Unique identifier for {col_name}"
        elif any(word in col_lower for word in ['date', 'time', 'created', 'updated']):
            return f"Date/time information for {col_name}"
        elif any(word in col_lower for word in ['name', 'title', 'label']):
            return f"Descriptive text field for {col_name}"
        elif any(word in col_lower for word in ['count', 'number', 'quantity', 'amount']):
            return f"Numeric count or quantity for {col_name}"
        elif any(word in col_lower for word in ['price', 'cost', 'value', 'amount']):
            return f"Monetary or numeric value for {col_name}"
        elif any(word in col_lower for word in ['status', 'state', 'flag']):
            return f"Status or state indicator for {col_name}"
        elif any(word in col_lower for word in ['email', 'mail']):
            return f"Email address for {col_name}"
        elif any(word in col_lower for word in ['phone', 'tel']):
            return f"Phone number for {col_name}"
        else:
            return f"Data field for {col_name}"
    
    def _get_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Get data quality metrics."""
        return {
            "completeness_score": (1 - df.isnull().sum().sum() / df.size) * 100,
            "uniqueness_score": (1 - df.duplicated().sum() / len(df)) * 100,
            "consistency_score": self._calculate_consistency_score(df),
            "overall_quality_score": 0  # Will be calculated
        }
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate consistency score based on data type consistency."""
        consistency_issues = 0
        total_checks = 0
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed data types in object columns
                total_checks += 1
                try:
                    # Try to convert to numeric
                    pd.to_numeric(df[col], errors='raise')
                except:
                    # Check if it's consistently non-numeric
                    if df[col].str.contains(r'^\d+$', na=False).any():
                        consistency_issues += 1
                else:
                    # It's numeric, check if it should be numeric
                    if not df[col].str.match(r'^\d+\.?\d*$', na=True).all():
                        consistency_issues += 1
        
        return (1 - consistency_issues / max(total_checks, 1)) * 100
    
    def _get_documentation_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Get recommendations for data documentation."""
        recommendations = []
        
        # Check for missing documentation
        if df.isnull().sum().sum() > 0:
            recommendations.append("Document missing value handling strategy")
        
        # Check for data type issues
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if df[col].str.contains(r'^\d+$', na=False).any():
                recommendations.append(f"Consider converting {col} to numeric type")
        
        # Check for potential sensitive data
        sensitive_patterns = ['ssn', 'social', 'credit', 'card', 'password', 'secret']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in sensitive_patterns):
                recommendations.append(f"Review {col} for sensitive data handling")
        
        return recommendations
    
    def _create_markdown_documentation(self, doc_report: Dict) -> str:
        """Create markdown documentation."""
        md = "# Data Documentation Report\n\n"
        md += f"**Generated:** {doc_report['file_info']['creation_timestamp']}\n\n"
        
        # File Information
        md += "## File Information\n\n"
        md += f"- **File Path:** {doc_report['file_info']['file_path']}\n"
        md += f"- **File Name:** {doc_report['file_info']['file_name']}\n"
        md += f"- **File Size:** {doc_report['file_info']['file_size_bytes']} bytes\n\n"
        
        # Data Summary
        md += "## Data Summary\n\n"
        md += f"- **Shape:** {doc_report['data_summary']['shape']}\n"
        md += f"- **Columns:** {len(doc_report['data_summary']['columns'])}\n"
        md += f"- **Memory Usage:** {doc_report['data_summary']['memory_usage']} bytes\n"
        md += f"- **Missing Values:** {sum(doc_report['data_summary']['missing_values'].values())}\n"
        md += f"- **Duplicate Rows:** {doc_report['data_summary']['duplicate_rows']}\n\n"
        
        # Data Dictionary
        md += "## Data Dictionary\n\n"
        md += "| Column | Type | Description | Missing % | Unique Values |\n"
        md += "|--------|------|-------------|-----------|---------------|\n"
        
        for col_info in doc_report['data_dictionary']:
            md += f"| {col_info['column_name']} | {col_info['data_type']} | {col_info['description']} | {col_info['missing_percentage']:.1f}% | {col_info['unique_values']} |\n"
        
        md += "\n"
        
        # Quality Metrics
        md += "## Quality Metrics\n\n"
        md += f"- **Completeness Score:** {doc_report['quality_metrics']['completeness_score']:.1f}%\n"
        md += f"- **Uniqueness Score:** {doc_report['quality_metrics']['uniqueness_score']:.1f}%\n"
        md += f"- **Consistency Score:** {doc_report['quality_metrics']['consistency_score']:.1f}%\n\n"
        
        # Recommendations
        if doc_report['recommendations']:
            md += "## Recommendations\n\n"
            for i, rec in enumerate(doc_report['recommendations'], 1):
                md += f"{i}. {rec}\n"
            md += "\n"
        
        return md

class ProcessLoggingToolInput(BaseModel):
    """Input schema for ProcessLoggingTool."""
    step_name: str = Field(..., description="Name of the processing step")
    step_description: str = Field(..., description="Description of what was done")
    input_file: str = Field(..., description="Input file path")
    output_file: str = Field(..., description="Output file path")
    parameters: Dict = Field(default={}, description="Parameters used in the step")
    success: bool = Field(default=True, description="Whether the step was successful")

class ProcessLoggingTool(BaseTool):
    name: str = "process_logging_tool"
    description: str = "Logs processing steps and maintains a processing history"
    args_schema: Type[BaseModel] = ProcessLoggingToolInput

    def _run(self, step_name: str, step_description: str, input_file: str, 
             output_file: str, parameters: Dict = {}, success: bool = True) -> str:
        """Log a processing step."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "step_name": step_name,
                "step_description": step_description,
                "input_file": input_file,
                "output_file": output_file,
                "parameters": parameters,
                "success": success
            }
            
            # Save to log file
            log_file = "processing_log.json"
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
            except FileNotFoundError:
                log_data = {"steps": []}
            
            log_data["steps"].append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            return f"Processing step logged successfully:\n" \
                   f"Step: {step_name}\n" \
                   f"Description: {step_description}\n" \
                   f"Input: {input_file}\n" \
                   f"Output: {output_file}\n" \
                   f"Success: {success}\n" \
                   f"Log saved to: {log_file}"
                   
        except Exception as e:
            return f"Error logging processing step: {str(e)}"
