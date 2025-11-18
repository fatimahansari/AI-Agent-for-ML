"""
Deployment training script - Train best performing model on full dataset and save as PKL file.
Generated automatically for production deployment.

This script:
1. Loads the full dataset (no train/test split)
2. Trains the best performing model on complete data
3. Saves trained model as PKL file for deployment
4. Also saves scaler if needed (for models requiring feature scaling)
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


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


def train_and_save_model(model_name: str, model_class, hyperparameters: dict, X, y, needs_scaling: bool, output_dir: str):
    """Train a single model on full dataset and save as PKL."""
    print(f"\n======================================================================")
    print(f"DEPLOYMENT TRAINING - {model_name} (Full Dataset)")
    print(f"======================================================================")
    
    # Feature scaling if needed
    scaler = None
    X_scaled = X
    if needs_scaling:
        print("Applying feature scaling...")
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    
    # Encode target labels to start from 0 (required for XGBoost/LightGBM)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = None
    y_encoded = y
    model_name_lower = model_name.lower()
    if 'xgboost' in model_name_lower or 'lightgbm' in model_name_lower:
        unique_classes = sorted(y.unique())
        if unique_classes[0] != 0:
            print("Encoding target labels to start from 0...")
            label_encoder = LabelEncoder()
            y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index)
    
    # Initialize model
    print(f"Training {model_name} on full dataset...")
    model = model_class(**hyperparameters)
    model.fit(X_scaled, y_encoded)
    print(f"[OK] {model_name} trained successfully")
    
    # Save model
    safe_model_name = model_name.lower().replace(' ', '_').replace('-', '_')
    model_filename = f"{output_dir}/model_{safe_model_name}.pkl"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving model to {model_filename}...")
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"[OK] Model saved: {model_filename}")
    
    # Save scaler if used
    if scaler is not None:
        safe_model_name = model_name.lower().replace(' ', '_').replace('-', '_')
        scaler_filename = f"{output_dir}/scaler_{safe_model_name}.pkl"
        print(f"Saving scaler to {scaler_filename}...")
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"[OK] Scaler saved: {scaler_filename}")
    
    # Save label encoder if used (for XGBoost/LightGBM)
    if label_encoder is not None:
        safe_model_name = model_name.lower().replace(' ', '_').replace('-', '_')
        encoder_filename = f"{output_dir}/label_encoder_{safe_model_name}.pkl"
        print(f"Saving label encoder to {encoder_filename}...")
        with open(encoder_filename, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"[OK] Label encoder saved: {encoder_filename}")
    
    return model, scaler


def main():
    """Main execution function."""
    try:
        # Configuration
        data_path = r"C:\Users\Danish\Desktop\llm-orchestration-agent\examples\AAPL.csv"
        target_column = "Day"
        output_dir = "models"  # Directory to save PKL files
        
        print("="*70)
        print("DEPLOYMENT TRAINING - Full Dataset Training")
        print("="*70)
        
        # Load and prepare data
        X, y = load_and_prepare_data(data_path, target_column)
        
        # Train best performing model on full dataset and save PKL
        model_xgboost, scaler_xgboost = train_and_save_model(
            model_name="XGBoost",
            model_class=xgb.XGBClassifier,
            hyperparameters={"random_state": 42},
            X=X,
            y=y,
            needs_scaling=False,
            output_dir=output_dir,
        )
        
        print("\n" + "="*70)
        print("DEPLOYMENT TRAINING COMPLETE")
        print("="*70)
        print(f"Trained {'XGBoost'} on full dataset")
        print(f"Model saved in: {output_dir}/")
        safe_model_name = "XGBoost".lower().replace(' ', '_').replace('-', '_')
        print("\nSaved files:")
        print(f"  - model_{safe_model_name}.pkl")
        if False:
            print(f"  - scaler_{safe_model_name}.pkl")
        model_name_lower = "XGBoost".lower()
        if 'xgboost' in model_name_lower or 'lightgbm' in model_name_lower:
            print(f"  - label_encoder_{safe_model_name}.pkl")
        
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
