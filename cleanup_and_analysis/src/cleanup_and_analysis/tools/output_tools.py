from crewai.tools import BaseTool
from typing import Type, Dict, List, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
from datetime import datetime

class OutputToolInput(BaseModel):
    """Input schema for OutputTool."""
    file_path: str = Field(..., description="Path to the CSV file to display")
    output_type: str = Field(default="summary", description="Type of output: 'summary', 'table', 'chart', 'full'")
    chart_type: str = Field(default="auto", description="Type of chart: 'auto', 'bar', 'line', 'scatter', 'histogram', 'heatmap'")
    max_rows: int = Field(default=20, description="Maximum number of rows to display in tables")

class OutputTool(BaseTool):
    name: str = "output_tool"
    description: str = "Displays data in terminal with tables, charts, and formatted output"
    args_schema: Type[BaseModel] = OutputToolInput

    def _run(self, file_path: str, output_type: str = "summary", 
             chart_type: str = "auto", max_rows: int = 20) -> str:
        """Display data in terminal with various output formats."""
        try:
            df = pd.read_csv(file_path)
            
            if output_type == "summary":
                return self._display_summary(df)
            elif output_type == "table":
                return self._display_table(df, max_rows)
            elif output_type == "chart":
                return self._display_chart(df, chart_type)
            elif output_type == "full":
                return self._display_full_output(df, max_rows, chart_type)
            else:
                return f"Unknown output type: {output_type}"
                
        except Exception as e:
            return f"Error displaying output: {str(e)}"
    
    def _display_summary(self, df: pd.DataFrame) -> str:
        """Display a summary of the dataset."""
        output = []
        output.append("=" * 60)
        output.append("DATASET SUMMARY")
        output.append("=" * 60)
        output.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        output.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        output.append("")
        
        # Data types summary
        output.append("DATA TYPES:")
        output.append("-" * 20)
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            output.append(f"{dtype}: {count} columns")
        output.append("")
        
        # Missing values summary
        missing = df.isnull().sum()
        if missing.sum() > 0:
            output.append("MISSING VALUES:")
            output.append("-" * 20)
            for col, count in missing[missing > 0].items():
                percentage = (count / len(df)) * 100
                output.append(f"{col}: {count} ({percentage:.1f}%)")
        else:
            output.append("MISSING VALUES: None")
        output.append("")
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        output.append(f"DUPLICATE ROWS: {duplicates}")
        output.append("")
        
        # Column information
        output.append("COLUMNS:")
        output.append("-" * 20)
        for col in df.columns:
            dtype = df[col].dtype
            unique = df[col].nunique()
            output.append(f"{col} ({dtype}) - {unique} unique values")
        
        return "\n".join(output)
    
    def _display_table(self, df: pd.DataFrame, max_rows: int = 20) -> str:
        """Display data in a formatted table."""
        output = []
        output.append("=" * 80)
        output.append("DATA TABLE")
        output.append("=" * 80)
        
        # Show first few rows
        display_df = df.head(max_rows)
        table_str = tabulate(display_df, headers='keys', tablefmt='grid', showindex=True)
        output.append(table_str)
        
        if len(df) > max_rows:
            output.append(f"\n... and {len(df) - max_rows} more rows")
        
        # Show basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            output.append("\n" + "=" * 80)
            output.append("NUMERIC STATISTICS")
            output.append("=" * 80)
            stats_df = df[numeric_cols].describe()
            stats_table = tabulate(stats_df, headers='keys', tablefmt='grid')
            output.append(stats_table)
        
        return "\n".join(output)
    
    def _display_chart(self, df: pd.DataFrame, chart_type: str = "auto") -> str:
        """Display charts and save them as images."""
        output = []
        output.append("=" * 60)
        output.append("GENERATING CHARTS")
        output.append("=" * 60)
        
        # Set up matplotlib for better display
        plt.style.use('default')
        sns.set_palette("husl")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        charts_created = []
        
        try:
            # Auto-determine chart type if needed
            if chart_type == "auto":
                if len(numeric_cols) >= 2:
                    chart_type = "scatter"
                elif len(numeric_cols) >= 1:
                    chart_type = "histogram"
                elif len(categorical_cols) >= 1:
                    chart_type = "bar"
                else:
                    return "No suitable data for charting found"
            
            # Create charts based on type
            if chart_type == "histogram" and len(numeric_cols) > 0:
                charts_created.extend(self._create_histograms(df, numeric_cols))
            elif chart_type == "bar" and len(categorical_cols) > 0:
                charts_created.extend(self._create_bar_charts(df, categorical_cols))
            elif chart_type == "scatter" and len(numeric_cols) >= 2:
                charts_created.extend(self._create_scatter_plots(df, numeric_cols))
            elif chart_type == "heatmap" and len(numeric_cols) > 1:
                charts_created.extend(self._create_heatmap(df, numeric_cols))
            elif chart_type == "line" and len(numeric_cols) > 0:
                charts_created.extend(self._create_line_plots(df, numeric_cols))
            
            output.append(f"Created {len(charts_created)} chart(s):")
            for chart_path in charts_created:
                output.append(f"  - {chart_path}")
            
        except Exception as e:
            output.append(f"Error creating charts: {str(e)}")
        
        return "\n".join(output)
    
    def _create_histograms(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        """Create histogram charts for numeric columns."""
        charts = []
        for i, col in enumerate(numeric_cols[:4]):  # Limit to 4 columns
            plt.figure(figsize=(10, 6))
            plt.hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            chart_path = f"charts/histogram_{col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts.append(chart_path)
        
        return charts
    
    def _create_bar_charts(self, df: pd.DataFrame, categorical_cols: List[str]) -> List[str]:
        """Create bar charts for categorical columns."""
        charts = []
        for i, col in enumerate(categorical_cols[:4]):  # Limit to 4 columns
            value_counts = df[col].value_counts().head(10)  # Top 10 values
            
            plt.figure(figsize=(12, 6))
            value_counts.plot(kind='bar')
            plt.title(f'Value Counts for {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            chart_path = f"charts/bar_{col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts.append(chart_path)
        
        return charts
    
    def _create_scatter_plots(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        """Create scatter plots for numeric columns."""
        charts = []
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
            plt.title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
            plt.xlabel(numeric_cols[0])
            plt.ylabel(numeric_cols[1])
            plt.grid(True, alpha=0.3)
            
            chart_path = f"charts/scatter_{numeric_cols[0]}_vs_{numeric_cols[1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts.append(chart_path)
        
        return charts
    
    def _create_heatmap(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        """Create correlation heatmap for numeric columns."""
        charts = []
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap')
            
            chart_path = f"charts/heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts.append(chart_path)
        
        return charts
    
    def _create_line_plots(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        """Create line plots for numeric columns."""
        charts = []
        for i, col in enumerate(numeric_cols[:3]):  # Limit to 3 columns
            plt.figure(figsize=(12, 6))
            plt.plot(df[col].dropna())
            plt.title(f'Line Plot of {col}')
            plt.xlabel('Index')
            plt.ylabel(col)
            plt.grid(True, alpha=0.3)
            
            chart_path = f"charts/line_{col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts.append(chart_path)
        
        return charts
    
    def _display_full_output(self, df: pd.DataFrame, max_rows: int = 20, chart_type: str = "auto") -> str:
        """Display comprehensive output including summary, table, and charts."""
        output = []
        
        # Summary
        output.append(self._display_summary(df))
        output.append("\n")
        
        # Table
        output.append(self._display_table(df, max_rows))
        output.append("\n")
        
        # Charts
        output.append(self._display_chart(df, chart_type))
        
        return "\n".join(output)

class TerminalDisplayToolInput(BaseModel):
    """Input schema for TerminalDisplayTool."""
    message: str = Field(..., description="Message to display in terminal")
    message_type: str = Field(default="info", description="Type of message: 'info', 'success', 'warning', 'error'")
    show_timestamp: bool = Field(default=True, description="Whether to show timestamp")

class TerminalDisplayTool(BaseTool):
    name: str = "terminal_display_tool"
    description: str = "Displays formatted messages in the terminal with different styles"
    args_schema: Type[BaseModel] = TerminalDisplayToolInput

    def _run(self, message: str, message_type: str = "info", show_timestamp: bool = True) -> str:
        """Display formatted message in terminal."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if show_timestamp else ""
            
            # Color codes for different message types
            colors = {
                "info": "\033[94m",      # Blue
                "success": "\033[92m",   # Green
                "warning": "\033[93m",   # Yellow
                "error": "\033[91m"      # Red
            }
            
            reset_color = "\033[0m"
            color = colors.get(message_type, colors["info"])
            
            # Format the message
            if show_timestamp:
                formatted_message = f"{color}[{timestamp}] {message}{reset_color}"
            else:
                formatted_message = f"{color}{message}{reset_color}"
            
            print(formatted_message)
            return f"Displayed {message_type} message: {message}"
            
        except Exception as e:
            return f"Error displaying message: {str(e)}"

class ProgressBarToolInput(BaseModel):
    """Input schema for ProgressBarTool."""
    current: int = Field(..., description="Current progress value")
    total: int = Field(..., description="Total progress value")
    description: str = Field(default="Processing", description="Description of the progress")

class ProgressBarTool(BaseTool):
    name: str = "progress_bar_tool"
    description: str = "Displays a progress bar in the terminal"
    args_schema: Type[BaseModel] = ProgressBarToolInput

    def _run(self, current: int, total: int, description: str = "Processing") -> str:
        """Display progress bar in terminal."""
        try:
            if total == 0:
                return "Cannot display progress bar: total is 0"
            
            percentage = (current / total) * 100
            bar_length = 50
            filled_length = int(bar_length * current // total)
            
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            progress_message = f'\r{description}: |{bar}| {current}/{total} ({percentage:.1f}%)'
            print(progress_message, end='', flush=True)
            
            if current == total:
                print()  # New line when complete
            
            return f"Progress: {current}/{total} ({percentage:.1f}%)"
            
        except Exception as e:
            return f"Error displaying progress bar: {str(e)}"
