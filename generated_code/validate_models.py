"""
Validation testing script - Validate models with 80-20 train/test split.
Generated automatically for model validation.

This script:
1. Loads and preprocesses the dataset
2. Splits data into 80% train and 20% test
3. Validates all selected models on test set
4. Calculates and displays evaluation metrics
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def load_and_prepare_data(data_path: str, target_column: str):
    """Load dataset and prepare for training."""
    # Load data
    data_path = os.path.abspath(data_path)
    print(f"Loading data from {data_path}...")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    print(f"Dataset shape: {df.shape}")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        print("Handling missing values...")
        # Numeric: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        # Categorical: fill with mode
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != target_column:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
                else:
                    df[col] = df[col].fillna('missing')
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode categorical variables (one-hot encoding)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"Encoding {len(categorical_cols)} categorical features...")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    print(f"Final feature shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def validate_model(model_name: str, model_class, hyperparameters: dict, X, y, needs_scaling: bool, test_size: float = 0.2, problem_type: str = "classification"):
    """Validate model with 80-20 train/test split and return metrics."""
    print(f"\n======================================================================")
    print(f"VALIDATION TESTING - {model_name} (80-20 Split)")
    print(f"======================================================================")
    
    # Split data (80-20)
    print(f"Splitting data: {100*(1-test_size):.0f}% train, {test_size*100:.0f}% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if problem_type == "classification" else None
    )
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Feature scaling if needed
    scaler = None
    X_train_scaled = X_train
    X_test_scaled = X_test
    if needs_scaling:
        print("Applying feature scaling...")
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    
    # Encode target labels to start from 0 (required for XGBoost/LightGBM)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = None
    y_train_encoded = y_train
    y_test_encoded = y_test
    model_name_lower = model_name.lower()
    if 'xgboost' in model_name_lower or 'lightgbm' in model_name_lower:
        unique_classes = sorted(y_train.unique())
        if unique_classes[0] != 0:
            print("Encoding target labels to start from 0...")
            label_encoder = LabelEncoder()
            y_train_encoded = pd.Series(label_encoder.fit_transform(y_train), index=y_train.index)
            y_test_encoded = pd.Series(label_encoder.transform(y_test), index=y_test.index)
    
    # Initialize and train model
    print(f"Training {model_name} on training set...")
    model = model_class(**hyperparameters)
    model.fit(X_train_scaled, y_train_encoded)
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = {"model_name": model_name}
    
    if problem_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_test_encoded, y_pred))
        metrics["precision"] = float(precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0))
        metrics["recall"] = float(recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0))
        metrics["f1_score"] = float(f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0))
        
        # Try ROC-AUC
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_scaled)
                if len(np.unique(y_test_encoded)) == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test_encoded, y_proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_test_encoded, y_proba, multi_class='ovr', average='weighted'))
            else:
                metrics["roc_auc"] = None
        except Exception:
            metrics["roc_auc"] = None
        
        print(f"\nVALIDATION METRICS - {model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision (weighted): {metrics['precision']:.4f}")
        print(f"  Recall (weighted): {metrics['recall']:.4f}")
        print(f"  F1-Score (weighted): {metrics['f1_score']:.4f}")
        if metrics.get('roc_auc') is not None:
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    else:  # regression
        metrics["mse"] = float(mean_squared_error(y_test_encoded, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test_encoded, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_test_encoded, y_pred))
        metrics["r2"] = float(r2_score(y_test_encoded, y_pred))
        
        print(f"\nVALIDATION METRICS - {model_name}:")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    return metrics


def main():
    """Main execution function."""
    try:
        # Configuration
        data_path = r"C:\Users\Danish\Desktop\llm-orchestration-agent\examples\AAPL.csv"
        target_column = "Day"
        test_size = 0.2  # 80-20 split for validation
        problem_type = "classification"
        
        print("="*70)
        print("MODEL VALIDATION TESTING (80-20 Split)")
        print("="*70)
        
        # Load and prepare data
        X, y = load_and_prepare_data(data_path, target_column)
        
        # Validate all models
        print("\n" + "="*70)
        print("VALIDATION TESTING - All Models")
        print("="*70)
        validation_results = []

        # Validate Random Forest with 80-20 split
        metrics_random_forest = validate_model(
            model_name="Random Forest",
            model_class=RandomForestClassifier,
            hyperparameters={"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "random_state": 42},
            X=X,
            y=y,
            needs_scaling=False,
            test_size=test_size,
            problem_type=problem_type,
        )
        validation_results.append(metrics_random_forest)

        # Validate XGBoost with 80-20 split
        metrics_xgboost = validate_model(
            model_name="XGBoost",
            model_class=xgb.XGBClassifier,
            hyperparameters={"random_state": 42},
            X=X,
            y=y,
            needs_scaling=False,
            test_size=test_size,
            problem_type=problem_type,
        )
        validation_results.append(metrics_xgboost)

        # Display validation results summary
        print("\n" + "="*70)
        print("VALIDATION TESTING RESULTS SUMMARY (80-20 Split)")
        print("="*70)
        for metrics in validation_results:
            model_name = metrics["model_name"]
            print(f"\n{model_name}:")
            if problem_type == "classification":
                print(f"  Accuracy: {{metrics.get('accuracy', 'N/A'):.4f}}" if 'accuracy' in metrics else "  Accuracy: N/A")
                print(f"  Precision: {{metrics.get('precision', 'N/A'):.4f}}" if 'precision' in metrics else "  Precision: N/A")
                print(f"  Recall: {{metrics.get('recall', 'N/A'):.4f}}" if 'recall' in metrics else "  Recall: N/A")
                print(f"  F1-Score: {{metrics.get('f1_score', 'N/A'):.4f}}" if 'f1_score' in metrics else "  F1-Score: N/A")
                if metrics.get('roc_auc') is not None:
                    print(f"  ROC-AUC: {{metrics['roc_auc']:.4f}}")
            else:
                print(f"  MSE: {{metrics.get('mse', 'N/A'):.4f}}" if 'mse' in metrics else "  MSE: N/A")
                print(f"  RMSE: {{metrics.get('rmse', 'N/A'):.4f}}" if 'rmse' in metrics else "  RMSE: N/A")
                print(f"  MAE: {{metrics.get('mae', 'N/A'):.4f}}" if 'mae' in metrics else "  MAE: N/A")
                print(f"  R²: {{metrics.get('r2', 'N/A'):.4f}}" if 'r2' in metrics else "  R²: N/A")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
