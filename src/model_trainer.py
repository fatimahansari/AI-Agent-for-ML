from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from src.model_recommender import DatasetMetadata, ModelRecommendation


class ModelTrainer:
    """Train and evaluate ML models on datasets."""

    def __init__(self, data_path: Path, metadata: DatasetMetadata, test_size: float = 0.2, random_state: int = 42):
        """Initialize trainer with dataset path and metadata."""
        self.data_path = data_path
        self.metadata = metadata
        self.test_size = test_size
        self.random_state = random_state
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.trained_models: Dict[str, Any] = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load dataset and return features and target."""
        # Try to load as CSV first, then other formats
        if self.data_path.suffix.lower() == '.csv':
            df = pd.read_csv(self.data_path)
        elif self.data_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(self.data_path)
        elif self.data_path.suffix.lower() == '.json':
            df = pd.read_json(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        if self.metadata.target not in df.columns:
            raise ValueError(f"Target column '{self.metadata.target}' not found in dataset")
        
        X = df.drop(columns=[self.metadata.target])
        y = df[self.metadata.target]
        
        return X, y

    def prepare_data(self) -> None:
        """Load and split data into train/test sets."""
        X, y = self.load_data()
        
        # Handle missing values (simple imputation for now)
        if X.isnull().sum().sum() > 0:
            # Numeric: fill with median
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
            # Categorical: fill with mode
            categorical_cols = X.select_dtypes(include=['object']).columns
            X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0] if len(X[categorical_cols].mode()) > 0 else 'missing')
        
        # Encode categorical variables (simple one-hot encoding)
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y if self.metadata.problem_type == "classification" else None
        )

    def instantiate_model(self, recommendation: ModelRecommendation) -> Any:
        """Create model instance from recommendation."""
        try:
            if recommendation.library == "sklearn":
                from sklearn import ensemble, linear_model, neighbors, naive_bayes, svm, tree
                module = sys.modules.get('sklearn')
            elif recommendation.library == "xgboost":
                import xgboost as xgb
                module = xgb
            elif recommendation.library == "lightgbm":
                import lightgbm as lgb
                module = lgb
            else:
                # Try to import as a module
                module = importlib.import_module(recommendation.library)
            
            model_class = getattr(module, recommendation.class_name)
            model = model_class(**recommendation.hyperparameters)
            return model
        except Exception as e:
            raise ValueError(f"Failed to instantiate {recommendation.library}.{recommendation.class_name}: {e}")

    def train_model(self, recommendation: ModelRecommendation) -> Any:
        """Train a single model."""
        if self.X_train is None:
            self.prepare_data()
        
        model = self.instantiate_model(recommendation)
        model.fit(self.X_train, self.y_train)
        self.trained_models[recommendation.name] = model
        return model

    def evaluate_model(self, model: Any, model_name: str) -> Dict[str, Any]:
        """Evaluate a trained model and return metrics."""
        y_pred = model.predict(self.X_test)
        
        metrics = {
            "model_name": model_name,
        }
        
        if self.metadata.problem_type == "classification":
            metrics.update({
                "accuracy": float(accuracy_score(self.y_test, y_pred)),
                "precision": float(precision_score(self.y_test, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(self.y_test, y_pred, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(self.y_test, y_pred, average='weighted', zero_division=0)),
            })
            
            # Try ROC-AUC (may fail for multiclass or non-binary)
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(self.X_test)
                    if len(np.unique(self.y_test)) == 2:
                        metrics["roc_auc"] = float(roc_auc_score(self.y_test, y_proba[:, 1]))
                    else:
                        metrics["roc_auc"] = float(roc_auc_score(self.y_test, y_proba, multi_class='ovr', average='weighted'))
            except Exception:
                metrics["roc_auc"] = None
            
            metrics["confusion_matrix"] = confusion_matrix(self.y_test, y_pred).tolist()
            metrics["classification_report"] = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
            
        elif self.metadata.problem_type == "regression":
            metrics.update({
                "mse": float(mean_squared_error(self.y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(self.y_test, y_pred))),
                "mae": float(mean_absolute_error(self.y_test, y_pred)),
                "r2": float(r2_score(self.y_test, y_pred)),
            })
        
        return metrics

    def train_and_evaluate(self, recommendations: List[ModelRecommendation]) -> List[Dict[str, Any]]:
        """Train multiple models and return evaluation results."""
        if self.X_train is None:
            self.prepare_data()
        
        results = []
        for recommendation in recommendations:
            try:
                model = self.train_model(recommendation)
                metrics = self.evaluate_model(model, recommendation.name)
                results.append(metrics)
            except Exception as e:
                results.append({
                    "model_name": recommendation.name,
                    "error": str(e),
                })
        
        return results

