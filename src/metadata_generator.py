"""Generate dataset metadata from data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.model_recommender import DatasetMetadata


def generate_metadata_from_file(
    data_path: Path,
    output_path: Optional[Path] = None,
    name: Optional[str] = None,
    target: Optional[str] = None,
    problem_type: Optional[str] = None,
    domain: Optional[str] = None,
    constraints: Optional[List[str]] = None,
    evaluation_metric: Optional[str] = None,
    notes: Optional[str] = None,
) -> tuple[DatasetMetadata, Path]:
    """Generate metadata JSON from a dataset file.
    
    Args:
        data_path: Path to the dataset file (CSV, Parquet, or JSON)
        output_path: Where to save metadata JSON (default: <dataset_name>_metadata.json)
        name: Custom dataset name
        target: Target column name (auto-detected if not specified)
        problem_type: Problem type (auto-detected if not specified)
        domain: Domain context
        constraints: List of constraints
        evaluation_metric: Preferred evaluation metric
        notes: Additional notes
        
    Returns:
        Tuple of (DatasetMetadata, output_path)
    """
    # Load dataset
    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix.lower() == ".json":
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    # Auto-detect target if not provided
    if target is None:
        target = _auto_detect_target(df)
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset")
    
    # Auto-detect problem type if not provided
    if problem_type is None:
        problem_type = _auto_detect_problem_type(df, target)
    
    # Analyze features
    feature_types = _classify_features(df, target)
    text_fields = [col for col, dtype in feature_types.items() if dtype == "text"]
    time_index = _detect_time_index(df)
    
    # Analyze target distribution
    target_dist = _analyze_target_distribution(df, target, problem_type)
    
    # Analyze missing values
    missing_values = _analyze_missing_values(df, target)
    
    # Auto-detect domain if not provided
    if domain is None:
        domain = _infer_domain(df.columns.tolist())
    
    # Default evaluation metric if not provided
    if evaluation_metric is None:
        if problem_type == "classification":
            evaluation_metric = "accuracy"
        elif problem_type == "regression":
            evaluation_metric = "rmse"
        else:
            evaluation_metric = "default"
    
    # Create metadata
    metadata = DatasetMetadata(
        name=name or data_path.stem,
        problem_type=problem_type,
        target=target,
        num_samples=len(df),
        num_features=len(df.columns) - 1,  # Exclude target
        feature_types=feature_types,
        target_distribution=target_dist,
        missing_values=missing_values,
        leakage_risks=None,
        text_fields=text_fields,
        time_index=time_index,
        constraints=constraints or [],
        evaluation_metric=evaluation_metric,
        domain=domain,
        notes=notes,
    )
    
    # Save to file
    if output_path is None:
        output_path = data_path.parent / f"{data_path.stem}_metadata.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    
    return metadata, output_path


def _auto_detect_target(df: pd.DataFrame) -> str:
    """Auto-detect target column from common patterns."""
    target_patterns = ["target", "label", "y", "churned", "churn", "forecast", "prediction"]
    
    for pattern in target_patterns:
        for col in df.columns:
            if pattern.lower() in col.lower():
                return col
    
    # If no pattern matches, return last column as fallback
    return df.columns[-1]


def _auto_detect_problem_type(df: pd.DataFrame, target: str) -> str:
    """Auto-detect problem type from target characteristics."""
    target_data = df[target]
    
    # Check if target is numeric
    if not pd.api.types.is_numeric_dtype(target_data):
        return "classification"
    
    # Check unique values ratio
    unique_ratio = target_data.nunique() / len(target_data)
    
    if unique_ratio < 0.1:  # Likely classification
        return "classification"
    elif unique_ratio > 0.9:  # Likely regression
        return "regression"
    else:
        # Check if values are mostly integers
        if target_data.dtype in ["int64", "int32"] and unique_ratio < 0.5:
            return "classification"
        else:
            return "regression"


def _classify_features(df: pd.DataFrame, target: str) -> dict[str, str]:
    """Classify feature types."""
    feature_types = {}
    
    for col in df.columns:
        if col == target:
            continue
            
        dtype = str(df[col].dtype)
        
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            feature_types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() / len(df) < 0.1:
                feature_types[col] = "categorical"
            else:
                feature_types[col] = "numeric"
        elif dtype == "object":
            # Check if it's text (long strings) or categorical
            sample_values = df[col].dropna().head(10)
            avg_length = sample_values.astype(str).str.len().mean()
            
            if avg_length > 20:  # Likely text
                feature_types[col] = "text"
            else:
                feature_types[col] = "categorical"
        else:
            feature_types[col] = "categorical"
    
    return feature_types


def _detect_time_index(df: pd.DataFrame) -> Optional[str]:
    """Detect time index column."""
    time_patterns = ["date", "time", "timestamp", "datetime"]
    
    for pattern in time_patterns:
        for col in df.columns:
            if pattern.lower() in col.lower():
                return col
    
    # Check if any datetime columns exist
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    return None


def _analyze_target_distribution(df: pd.DataFrame, target: str, problem_type: str) -> str:
    """Analyze target variable distribution."""
    target_data = df[target].dropna()
    
    if problem_type == "classification":
        class_counts = target_data.value_counts()
        if len(class_counts) == 2:
            minority_ratio = class_counts.min() / class_counts.sum()
            return f"Binary target with class imbalance - minority class at {minority_ratio:.1%} positive class ratio"
        else:
            return f"Multi-class target with {len(class_counts)} classes"
    else:  # regression
        mean = target_data.mean()
        std = target_data.std()
        min_val = target_data.min()
        max_val = target_data.max()
        median = target_data.median()
        return f"Continuous target: mean={mean:.2f}, std={std:.2f}, range=[{min_val:.2f}, {max_val:.2f}], median={median:.2f}"


def _analyze_missing_values(df: pd.DataFrame, target: str) -> Optional[str]:
    """Analyze missing value patterns."""
    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    
    if len(missing_cols) == 0:
        return None
    
    missing_info = []
    for col, count in missing_cols.items():
        if col == target:
            continue
        pct = (count / len(df)) * 100
        missing_info.append(f"{col} has {pct:.1f}% missing values")
    
    if not missing_info:
        return None
    
    return "Sparse missingness: " + ", ".join(missing_info)


def _infer_domain(columns: List[str]) -> Optional[str]:
    """Infer domain from column names."""
    column_str = " ".join(columns).lower()
    
    domain_keywords = {
        "finance": ["price", "stock", "volume", "close", "open", "high", "low", "ticker", "dividend"],
        "telecom": ["churn", "subscription", "plan", "contract", "tenure", "support"],
        "healthcare": ["patient", "diagnosis", "treatment", "medication", "hospital"],
        "ecommerce": ["purchase", "cart", "product", "review", "rating"],
        "marketing": ["campaign", "conversion", "click", "impression", "ad"],
    }
    
    for domain, keywords in domain_keywords.items():
        if any(keyword in column_str for keyword in keywords):
            return domain
    
    return None
