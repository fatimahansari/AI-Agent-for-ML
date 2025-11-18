"""Model compatibility checker to filter models that cannot work with the dataset."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from src.model_recommender import DatasetMetadata


class ModelCompatibilityChecker:
    """Check if models are compatible with dataset characteristics."""
    
    # Model-specific constraints and their compatibility checks
    MODEL_CONSTRAINTS = {
        "svm": {
            "requires_multiple_classes": True,  # SVM requires >1 class
            "min_samples": 2,
        },
        "support vector": {
            "requires_multiple_classes": True,
            "min_samples": 2,
        },
        "neural network": {
            "requires_multiple_classes": True,
            "min_samples": 10,
        },
        "mlp": {
            "requires_multiple_classes": True,
            "min_samples": 10,
        },
        "logistic regression": {
            "requires_multiple_classes": True,  # For classification
            "min_samples": 2,
        },
        "random forest": {
            "min_samples": 2,
        },
        "xgboost": {
            "min_samples": 2,
        },
        "lightgbm": {
            "min_samples": 2,
        },
        "gradient boosting": {
            "min_samples": 2,
        },
    }
    
    def __init__(self, metadata: DatasetMetadata):
        """Initialize compatibility checker with dataset metadata.
        
        Args:
            metadata: Dataset metadata to check compatibility against
        """
        self.metadata = metadata
        self.num_classes = self._extract_num_classes()
    
    def _extract_num_classes(self) -> Optional[int]:
        """Extract number of classes from target distribution.
        
        Returns:
            Number of classes if classification, None otherwise
        """
        if self.metadata.problem_type != "classification":
            return None
        
        target_dist = self.metadata.target_distribution or ""
        target_dist_lower = target_dist.lower()
        
        # Look for patterns like "31 classes", "multi-class target with 31 classes", "binary target"
        if "binary" in target_dist_lower:
            return 2
        
        # Look for number patterns - try multiple patterns
        # "Multi-class target with 31 classes"
        class_match = re.search(r'(\d+)\s+class', target_dist_lower)
        if class_match:
            return int(class_match.group(1))
        
        # "31 classes" 
        class_match = re.search(r'(\d+)\s*class', target_dist_lower)
        if class_match:
            return int(class_match.group(1))
        
        # Just a number before "class" or "classes"
        class_match = re.search(r'(\d+)(?:\s+)?class', target_dist_lower)
        if class_match:
            return int(class_match.group(1))
        
        # Default: assume multiple classes if classification (we don't know exact number)
        return None
    
    def is_model_compatible(self, model_name: str) -> tuple[bool, Optional[str]]:
        """Check if a model is compatible with the dataset.
        
        Args:
            model_name: Name of the model to check (e.g., "Random Forest", "SVM")
            
        Returns:
            Tuple of (is_compatible, reason_if_incompatible)
        """
        model_lower = model_name.lower()
        
        # Check each constraint pattern
        for constraint_name, constraints in self.MODEL_CONSTRAINTS.items():
            if constraint_name in model_lower:
                # Check multiple classes requirement
                if constraints.get("requires_multiple_classes") and self.metadata.problem_type == "classification":
                    if self.num_classes is not None and self.num_classes < 2:
                        return False, f"{model_name} requires at least 2 classes, but dataset has only {self.num_classes} class"
                    # Also check if we can determine num_classes from metadata
                    if self.num_classes is None:
                        # We can't verify, but warn - actually allow it and let execution catch it
                        pass
                
                # Check minimum samples
                min_samples = constraints.get("min_samples", 0)
                if self.metadata.num_samples and self.metadata.num_samples < min_samples:
                    return False, f"{model_name} requires at least {min_samples} samples, but dataset has only {self.metadata.num_samples}"
        
        # Special checks for specific models
        if "svm" in model_lower or "support vector" in model_lower:
            # SVM requires >1 unique class in target (at least 2 classes)
            if self.metadata.problem_type == "classification":
                if self.num_classes is not None and self.num_classes < 2:
                    return False, "SVM requires at least 2 classes (got only 1 unique class)"
                # Note: SVM can fail at runtime if train/test split results in only 1 class in a set
                # This is checked at runtime, but we can still warn
        
        if "logistic regression" in model_lower:
            if self.metadata.problem_type == "classification":
                if self.num_classes is not None and self.num_classes < 2:
                    return False, "Logistic Regression for classification requires at least 2 classes"
        
        # Note: Some models can fail at runtime even if they pass static checks
        # (e.g., if train/test split results in only 1 class in training or test set)
        # These are caught during execution and handled separately
        
        # All checks passed
        return True, None
    
    def filter_compatible_models(self, model_names: List[str]) -> Dict[str, tuple[bool, Optional[str]]]:
        """Filter models and return compatibility status for each.
        
        Args:
            model_names: List of model names to check
            
        Returns:
            Dictionary mapping model_name -> (is_compatible, reason_if_incompatible)
        """
        compatibility = {}
        for model_name in model_names:
            is_compat, reason = self.is_model_compatible(model_name)
            compatibility[model_name] = (is_compat, reason)
        return compatibility
    
    def get_compatible_models(self, model_names: List[str]) -> List[str]:
        """Get only compatible models from a list.
        
        Args:
            model_names: List of model names to check
            
        Returns:
            List of compatible model names
        """
        compatible = []
        for model_name in model_names:
            is_compat, _ = self.is_model_compatible(model_name)
            if is_compat:
                compatible.append(model_name)
        return compatible

