from crewai.tools import BaseTool
from typing import Type, Dict, List, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class TransformationToolInput(BaseModel):
    """Input schema for TransformationTool."""
    file_path: str = Field(..., description="Path to the CSV file to transform")
    transformations: List[str] = Field(default=[], description="List of transformations to apply")
    groupby_columns: List[str] = Field(default=[], description="Columns to group by for aggregations")
    aggregation_functions: List[str] = Field(default=["mean", "sum", "count"], description="Aggregation functions to apply")

class TransformationTool(BaseTool):
    name: str = "transformation_tool"
    description: str = "Transforms data by creating derived fields, aggregations, and reformatting"
    args_schema: Type[BaseModel] = TransformationToolInput

    def _run(self, file_path: str, transformations: List[str] = [], 
             groupby_columns: List[str] = [], aggregation_functions: List[str] = ["mean", "sum", "count"]) -> str:
        """Transform CSV data and return transformation report."""
        try:
            # Flatten transformations list
            if transformations and isinstance(transformations[0], list):
                transformations = [item for sublist in transformations for item in sublist]

            resolved_path = self._resolve_latest_source(file_path)
            df = pd.read_csv(resolved_path)
            original_shape = df.shape
            transformation_report = {
                "original_shape": original_shape,
                "transformations_applied": [],
                "new_columns": [],
                "final_shape": None
            }
            
            # Apply automatic transformations
            df = self._apply_automatic_transformations(df, transformation_report)
            
            # Apply requested transformations
            for transformation in transformations:
                df = self._apply_transformation(df, transformation, transformation_report)
            
            # Apply aggregations if groupby columns specified
            if groupby_columns:
                df = self._apply_aggregations(df, groupby_columns, aggregation_functions, transformation_report)
            
            transformation_report["final_shape"] = df.shape
            
            # Save transformed data
            # Determine canonical output path under input/ using original base name
            base_name = os.path.basename(file_path).replace('.csv', '')
            output_path = os.path.join('input', f"{base_name}_transformed.csv")
            df.to_csv(output_path, index=False)
            
            return f"Data transformation completed:\n" \
                   f"Original shape: {transformation_report['original_shape']}\n" \
                   f"Transformations applied: {transformation_report['transformations_applied']}\n" \
                   f"New columns created: {transformation_report['new_columns']}\n" \
                   f"Final shape: {transformation_report['final_shape']}\n" \
                   f"Transformed data saved to: {output_path}"
                   
        except Exception as e:
            return f"Error transforming CSV file: {str(e)}"
    
    def _apply_automatic_transformations(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Apply automatic transformations based on data analysis."""
        # Create date-based features if date columns exist
        date_columns = df.select_dtypes(include=['datetime64[ns]']).columns
        for col in date_columns:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_weekday'] = df[col].dt.weekday
            report["new_columns"].extend([f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_weekday'])
            report["transformations_applied"].append(f"Created date features for {col}")
        
        # Create categorical features for object columns with few unique values
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() < 20 and df[col].nunique() > 1:
                df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
                report["new_columns"].append(f'{col}_encoded')
                report["transformations_applied"].append(f"Encoded categorical column {col}")
        
        return df
    
    def _apply_transformation(self, df: pd.DataFrame, transformation: str, report: Dict) -> pd.DataFrame:
        """Apply a specific transformation."""
        # Natural-language ratio creation: "Create ratio of A to B (name)"
        if isinstance(transformation, str) and transformation.lower().startswith("create ratio of "):
            try:
                # parse like: Create ratio of seedlings to nurseries (seedlings_to_nurseries)
                body = transformation[len("Create ratio of "):]
                parts = body.split(" to ")
                if len(parts) >= 2:
                    left = parts[0].strip()
                    right_part = parts[1]
                    # optional name in parentheses
                    if "(" in right_part and ")" in right_part:
                        right = right_part.split("(")[0].strip()
                        new_name = right_part[right_part.find("(")+1:right_part.rfind(")")] .strip()
                    else:
                        right = right_part.strip()
                        new_name = f"{left}_to_{right}_ratio"
                    if left in df.columns and right in df.columns:
                        denom = df[right].replace({0: np.nan})
                        df[new_name] = pd.to_numeric(df[left], errors='coerce') / pd.to_numeric(denom, errors='coerce')
                        report["new_columns"].append(new_name)
                        report["transformations_applied"].append(f"Created ratio {new_name} = {left}/{right}")
                        return df
            except Exception:
                return df

        if transformation == "normalize_numeric":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                report["new_columns"].append(f'{col}_normalized')
            report["transformations_applied"].append("Normalized numeric columns")
        
        elif transformation == "log_transform":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if (df[col] > 0).all():
                    df[f'{col}_log'] = np.log(df[col])
                    report["new_columns"].append(f'{col}_log')
            report["transformations_applied"].append("Applied log transformation to positive numeric columns")
        
        elif transformation == "create_ratios":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        if (df[col2] != 0).all():
                            df[f'{col1}_to_{col2}_ratio'] = df[col1] / df[col2]
                            report["new_columns"].append(f'{col1}_to_{col2}_ratio')
                report["transformations_applied"].append("Created ratio features")
        
        return df
    
    def _apply_aggregations(self, df: pd.DataFrame, groupby_columns: List[str], 
                          aggregation_functions: List[str], report: Dict) -> pd.DataFrame:
        """Apply aggregations to grouped data."""
        if not groupby_columns:
            return df
        
        # Ensure groupby columns exist
        valid_groupby = [col for col in groupby_columns if col in df.columns]
        if not valid_groupby:
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        agg_dict = {}
        
        for func in aggregation_functions:
            if func in ['mean', 'sum', 'count', 'min', 'max', 'std', 'median']:
                for col in numeric_cols:
                    agg_dict[col] = func
        
        if agg_dict:
            df_agg = df.groupby(valid_groupby).agg(agg_dict).reset_index()
            df_agg.columns = [f"{col}_{func}" if func in ['mean', 'sum', 'count', 'min', 'max', 'std', 'median'] 
                             else col for col, func in df_agg.columns]
            report["transformations_applied"].append(f"Applied aggregations grouped by {valid_groupby}")
            return df_agg
        
        return df

    def _resolve_latest_source(self, file_path: str) -> str:
        """Prefer cleaned/standardized/engineered versions if available in order."""
        base_name = os.path.basename(file_path).replace('.csv', '')
        candidates = [
            os.path.join('intermediate_csv', f"{base_name}_engineered.csv"),
            os.path.join('intermediate_csv', f"{base_name}_cleaned.csv"),
            os.path.join('intermediate_csv', f"{base_name}_standardized.csv"),
            file_path
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return file_path

class FeatureEngineeringToolInput(BaseModel):
    """Input schema for FeatureEngineeringTool."""
    file_path: str = Field(..., description="Path to the CSV file to engineer features for")
    target_column: Optional[str] = Field(default=None, description="Target column for feature engineering")

class FeatureEngineeringTool(BaseTool):
    name: str = "feature_engineering_tool"
    description: str = "Engineers new features to make data easier to analyze"
    args_schema: Type[BaseModel] = FeatureEngineeringToolInput

    def _run(self, file_path: str, target_column: Optional[str] = None) -> str:
        """Engineer features and return feature engineering report."""
        try:
            df = pd.read_csv(file_path)
            original_columns = list(df.columns)
            feature_report = {
                "original_columns": len(original_columns),
                "new_features": [],
                "feature_importance": {}
            }
            
            # Create interaction features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                        feature_report["new_features"].append(f'{col1}_x_{col2}')
            
            # Create polynomial features
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_cubed'] = df[col] ** 3
                feature_report["new_features"].extend([f'{col}_squared', f'{col}_cubed'])
            
            # Create statistical features
            for col in numeric_cols:
                df[f'{col}_rolling_mean_3'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_rolling_std_3'] = df[col].rolling(window=3, min_periods=1).std()
                feature_report["new_features"].extend([f'{col}_rolling_mean_3', f'{col}_rolling_std_3'])
            
            # Calculate feature importance if target column specified
            if target_column and target_column in df.columns:
                feature_report["feature_importance"] = self._calculate_feature_importance(df, target_column)
            
            # Save engineered data
            file_name = file_path.split('\\')[-1].split('/')[-1]
            output_path = f"intermediate_csv/{file_name.replace('.csv', '_engineered.csv')}"
            df.to_csv(output_path, index=False)
            
            return f"Feature engineering completed:\n" \
                   f"Original columns: {feature_report['original_columns']}\n" \
                   f"New features created: {len(feature_report['new_features'])}\n" \
                   f"New features: {feature_report['new_features']}\n" \
                   f"Feature importance: {feature_report['feature_importance']}\n" \
                   f"Engineered data saved to: {output_path}"
                   
        except Exception as e:
            return f"Error engineering features: {str(e)}"
    
    def _calculate_feature_importance(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Calculate feature importance using correlation."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        importance = {}
        
        for col in numeric_cols:
            if col != target_column:
                corr = df[col].corr(df[target_column])
                importance[col] = abs(corr) if not pd.isna(corr) else 0
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
