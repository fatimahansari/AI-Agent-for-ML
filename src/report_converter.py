"""Convert Pre Processing Agent report format to Reasoning Agent DatasetMetadata format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.model_recommender import DatasetMetadata


def convert_report_to_metadata(
    report_path: Path,
    dataset_name: Optional[str] = None,
    domain: Optional[str] = None,
    evaluation_metric: Optional[str] = None,
    constraints: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> DatasetMetadata:
    """Convert Pre Processing Agent report JSON to DatasetMetadata format.
    
    Args:
        report_path: Path to Pre Processing Agent report JSON file
        dataset_name: Optional dataset name (defaults to report filename without extension)
        domain: Optional domain context (e.g., "finance", "real-estate")
        evaluation_metric: Optional evaluation metric (auto-detected if not provided)
        constraints: Optional list of constraints
        notes: Optional additional notes
        
    Returns:
        DatasetMetadata object
    """
    # Load report
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    
    # Extract dataset name
    if dataset_name is None:
        dataset_name = report_path.stem
    
    # Extract basic info from report
    overview = report.get("dataset_overview", {})
    target_analysis = report.get("target_analysis", {})
    columns = report.get("columns", {})
    
    # Get target column
    target = overview.get("target_column")
    if not target:
        raise ValueError("Target column not found in report")
    
    # Determine problem type from target analysis
    target_type = target_analysis.get("type", "numeric")
    if target_type == "categorical":
        problem_type = "classification"
    elif target_type == "numeric":
        problem_type = "regression"
    else:
        problem_type = "classification"  # Default fallback
    
    # Get dataset size
    num_samples = overview.get("rows")
    num_features = overview.get("columns", 0) - 1  # Exclude target column
    
    # Build feature_types dictionary
    feature_types: Dict[str, str] = {}
    text_fields: List[str] = []
    time_index: Optional[str] = None
    
    for col_name, col_info in columns.items():
        if col_name == target:
            continue
        
        dtype = col_info.get("dtype", "object")
        
        # Classify feature type
        if dtype in ["int64", "float64", "int32", "float32"]:
            feature_types[col_name] = "numeric"
        elif dtype == "object" or dtype == "category":
            # Check if it's a date/time column
            if "date" in col_name.lower() or "time" in col_name.lower() or "timestamp" in col_name.lower():
                feature_types[col_name] = "datetime"
                if time_index is None:
                    time_index = col_name
            else:
                # Check unique values to determine if categorical or text
                unique_vals = col_info.get("unique_values", 0)
                if unique_vals > 50:  # High cardinality - likely text
                    feature_types[col_name] = "text"
                    text_fields.append(col_name)
                else:
                    feature_types[col_name] = "categorical"
        else:
            feature_types[col_name] = "categorical"
    
    # Build target distribution description
    target_distribution: Optional[str] = None
    if target_type == "categorical":
        unique_classes = target_analysis.get("unique_classes", 0)
        imbalance = target_analysis.get("class_imbalance_ratio", 1.0)
        if unique_classes == 2:
            target_distribution = f"Binary classification (imbalance ratio: {imbalance:.2f})"
        else:
            target_distribution = f"Multi-class target with {unique_classes} classes (imbalance ratio: {imbalance:.2f})"
    else:
        # Numeric target
        mean_val = target_analysis.get("mean")
        std_val = target_analysis.get("std")
        skew_val = target_analysis.get("skew", 0)
        if mean_val is not None and std_val is not None:
            if abs(skew_val) > 1:
                target_distribution = f"Numeric target (mean: {mean_val:.2f}, std: {std_val:.2f}, skew: {skew_val:.2f} - highly skewed)"
            else:
                target_distribution = f"Numeric target (mean: {mean_val:.2f}, std: {std_val:.2f})"
    
    # Analyze missing values
    missing_info: List[str] = []
    for col_name, col_info in columns.items():
        missing_pct = col_info.get("missing_percent", 0.0)
        if missing_pct > 0:
            missing_info.append(f"{col_name}: {missing_pct*100:.1f}%")
    
    missing_values: Optional[str] = None
    if missing_info:
        missing_values = f"Missing values detected: {', '.join(missing_info[:5])}"  # Limit to first 5
    else:
        missing_values = "No missing values detected"
    
    # Auto-detect evaluation metric if not provided
    if evaluation_metric is None:
        if problem_type == "classification":
            evaluation_metric = "accuracy"
        elif problem_type == "regression":
            evaluation_metric = "rmse"
        else:
            evaluation_metric = "default"
    
    # Extract high-level flags as constraints if not provided
    if constraints is None:
        constraints = []
        flags = report.get("llm_flags", [])
        high_level_flags = report.get("high_level_flags", [])
        
        # Convert flags to constraints
        if "target_imbalanced" in flags:
            constraints.append("Class imbalance detected - consider class weights or resampling")
        if "target_skewed" in flags:
            constraints.append("Target variable is highly skewed - consider log transformation")
        if "numeric_skewed_features" in flags:
            constraints.append("Some numeric features are highly skewed - consider transformation")
        if "numeric_outlier_features" in flags:
            constraints.append("Some features have high outlier fraction - consider robust scaling")
        if "high_missing_columns" in flags:
            constraints.append("Some columns have very high missing values (>30%)")
        if "high_cardinality_categorical" in flags:
            constraints.append("High-cardinality categorical features detected")
        
        # Add high-level flags as notes
        if high_level_flags and not notes:
            notes = "; ".join(high_level_flags[:3])  # Limit to first 3
    
    # Infer domain from column names if not provided
    if domain is None:
        domain = _infer_domain_from_columns(list(columns.keys()))
    
    # Create DatasetMetadata
    metadata = DatasetMetadata(
        name=dataset_name,
        problem_type=problem_type,
        target=target,
        num_samples=num_samples,
        num_features=num_features,
        feature_types=feature_types,
        target_distribution=target_distribution,
        missing_values=missing_values,
        leakage_risks=None,  # Could be enhanced to detect leakage risks
        text_fields=text_fields,
        time_index=time_index,
        constraints=constraints,
        evaluation_metric=evaluation_metric,
        domain=domain,
        notes=notes,
    )
    
    return metadata


def _infer_domain_from_columns(column_names: List[str]) -> Optional[str]:
    """Infer domain from column names."""
    column_names_lower = [col.lower() for col in column_names]
    
    # Finance domain indicators
    finance_keywords = ["price", "close", "open", "high", "low", "volume", "stock", "ticker", "market"]
    if any(keyword in " ".join(column_names_lower) for keyword in finance_keywords):
        return "finance"
    
    # Real estate domain indicators
    real_estate_keywords = ["price", "bedroom", "bathroom", "area", "sqft", "house", "property"]
    if any(keyword in " ".join(column_names_lower) for keyword in real_estate_keywords):
        return "real-estate"
    
    # Healthcare domain indicators
    healthcare_keywords = ["patient", "diagnosis", "treatment", "symptom", "disease"]
    if any(keyword in " ".join(column_names_lower) for keyword in healthcare_keywords):
        return "healthcare"
    
    # Telecom domain indicators
    telecom_keywords = ["churn", "customer", "contract", "phone", "plan"]
    if any(keyword in " ".join(column_names_lower) for keyword in telecom_keywords):
        return "telecom"
    
    # E-commerce domain indicators
    ecommerce_keywords = ["product", "order", "cart", "purchase", "customer"]
    if any(keyword in " ".join(column_names_lower) for keyword in ecommerce_keywords):
        return "e-commerce"
    
    return None
