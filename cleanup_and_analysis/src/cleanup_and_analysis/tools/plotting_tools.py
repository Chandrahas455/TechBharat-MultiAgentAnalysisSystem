from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt


class PlottingToolInput(BaseModel):
    """Input schema for PlottingTool."""
    file_path: str = Field(..., description="Path to the CSV file to read")
    x_column: str = Field(..., description="Numeric column for x-axis")
    y_column: str = Field(..., description="Numeric column for y-axis")
    title: Optional[str] = Field(default=None, description="Optional plot title")


class PlottingTool(BaseTool):
    name: str = "plotting_tool"
    description: str = "Generates a scatter plot for the given x and y numeric columns with a clear title."
    args_schema: Type[BaseModel] = PlottingToolInput

    def _run(self, file_path: str, x_column: str, y_column: str, title: Optional[str] = None) -> str:
        """Create a scatter plot for the requested pair if valid and return saved path."""
        try:
            resolved_path = self._resolve_latest_source(file_path)
            if not os.path.exists(resolved_path):
                return f"Error: File not found: {resolved_path}"

            df = pd.read_csv(resolved_path)
            if x_column not in df.columns or y_column not in df.columns:
                return f"Error: Columns not found. Available: {list(df.columns)}"

            # Ensure numeric
            if not np.issubdtype(df[x_column].dropna().dtype, np.number) or not np.issubdtype(df[y_column].dropna().dtype, np.number):
                # Try coerce
                df[x_column] = pd.to_numeric(df[x_column], errors='coerce')
                df[y_column] = pd.to_numeric(df[y_column], errors='coerce')

            x = df[x_column].astype(float)
            y = df[y_column].astype(float)
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                return "Error: Not enough valid numeric data points to plot."

            # Compute correlation for title context
            corr = float(x[valid].corr(y[valid])) if valid.sum() > 2 else float('nan')
            corr_text = f" (r={corr:.2f})" if not np.isnan(corr) else ""

            # Prepare title
            final_title = title if title else f"{x_column} vs {y_column}{corr_text}"

            os.makedirs("charts", exist_ok=True)
            plt.figure(figsize=(10, 7))
            plt.scatter(x[valid], y[valid], alpha=0.65, edgecolors='none')
            plt.title(final_title)
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.grid(True, alpha=0.25)

            chart_path = os.path.join(
                "charts",
                f"scatter_{x_column}_vs_{y_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"Saved plot: {chart_path}"
        except Exception as e:
            return f"Error generating plot: {str(e)}"

    def _resolve_latest_source(self, file_path: str) -> str:
        """Prefer latest transformed/engineered/cleaned/standardized versions if available."""
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


