"""Generate validation/testing code for models using LLM."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_ollama import OllamaLLM

from src.model_recommender import DatasetMetadata

DEFAULT_MODEL_NAME = "phi4"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TEMPERATURE = 0.3


class CodeGenerator:
    """Generate Python validation/testing code for ML models using LLM."""
    
    def __init__(self) -> None:
        self.llm = OllamaLLM(
            model=DEFAULT_MODEL_NAME,
            base_url=DEFAULT_BASE_URL,
            temperature=DEFAULT_TEMPERATURE,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert ML engineer. Generate complete, standalone Python code 
                    for training and validating a machine learning model compatible with scikit-learn 1.3+.
                    
                    NOTE: Dataset preprocessing is handled by a separate agent. Your code should focus ONLY on:
                    1. Loading the preprocessed/ready dataset
                    2. Basic data splitting
                    3. Feature scaling (if needed for the model)
                    4. Model training with appropriate hyperparameters
                    5. Model evaluation and metrics printing

                    The code must follow these steps:
                    
                    STEP 1: Load Dataset
                    - Load dataset using pandas: pd.read_csv(), pd.read_parquet(), or pd.read_json()
                    - Print dataset shape for verification
                    - Extract features (X) and target (y) based on the target column name
                    
                    STEP 2: Data Splitting
                    - Use train_test_split from sklearn.model_selection with random_state=42
                    - For classification: Use stratify=y if classes are reasonably balanced
                    - Set test_size as specified
                    
                    STEP 3: Feature Scaling (Only if Required)
                    - For models requiring scaling (Logistic Regression, SVM, Neural Networks):
                      * Use StandardScaler from sklearn.preprocessing
                      * Fit scaler ONLY on X_train, transform both X_train and X_test
                    - For tree-based models (Random Forest, XGBoost, LightGBM): NO scaling needed
                    
                    STEP 4: Model Training with Appropriate Hyperparameters
                    - Use model hyperparameters suitable for the problem type:
                      * Multi-class classification: Ensure multi_class parameter is set correctly
                      * For class imbalance: Consider class_weight='balanced' or sample_weight
                      * For Random Forest: Use n_estimators=200, max_depth=10-20, min_samples_split=5
                      * For XGBoost/LightGBM: Use appropriate objective for multi-class
                      * For Logistic Regression: max_iter=2000+, solver='lbfgs' or 'saga' for multi-class
                      * Set random_state=42 for reproducibility
                    
                    STEP 5: Evaluation and Metrics
                    - Evaluate and print metrics in EXACT format (colon and space required):
                      Classification: print("Accuracy:", value)
                                      print("Precision:", value)  # use average='weighted' for multi-class
                                      print("Recall:", value)     # use average='weighted' for multi-class
                                      print("F1-Score:", value)   # use average='weighted' for multi-class
                                      print("ROC-AUC:", value)    # use multi_class='ovr', average='weighted' for multi-class
                      Regression:     print("RMSE:", value)
                                      print("MSE:", value)
                                      print("MAE:", value)
                                      print("R²:", value)
                    - For multi-class problems: ALWAYS use average='weighted' for precision/recall/f1

                    CRITICAL API COMPATIBILITY (scikit-learn 1.3+):
                    - LogisticRegression: Set max_iter=2000 or higher to avoid convergence warnings
                      Example: LogisticRegression(max_iter=2000, random_state=42, multi_class='multinomial')
                    - Always use StandardScaler for Logistic Regression, SVM, and Neural Networks
                    - For multi-class ROC-AUC: roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')

                    CODE STRUCTURE REQUIREMENTS:
                    - Include ALL imports at the top
                    - Use absolute path for dataset: import os; dataset_path = os.path.abspath(r"<path>")
                    - Wrap main code in try-except for error handling
                    - Print error messages if something fails
                    - Code must run from any directory
                    - Use only: scikit-learn, pandas, numpy, os (standard library)
                    
                    MODEL-SPECIFIC NOTES:
                    - Random Forest: 
                      * Works well with mixed data types, no scaling needed
                      * Use n_estimators=200, max_depth=10-20, min_samples_split=5, min_samples_leaf=2
                      * For multi-class: Use criterion='gini' or 'entropy'
                      * Set class_weight='balanced' if class imbalance detected
                    - XGBoost/LightGBM: 
                      * Can handle categoricals, use native categorical support if available
                      * Use objective='multi:softprob' for multi-class, num_class=number_of_classes
                      * Set learning_rate=0.1, n_estimators=100-200, max_depth=5-7
                      * Use scale_pos_weight or class_weight for imbalance
                    - Logistic Regression: 
                      * MUST scale features (StandardScaler) 
                      * MUST set max_iter=2000+, solver='lbfgs' for small datasets or 'saga' for large
                      * For multi-class: Use multi_class='multinomial' or OneVsRestClassifier wrapper
                      * Set class_weight='balanced' for imbalanced data
                    - SVM: 
                      * MUST scale features
                      * Use SVC with kernel='rbf', C=1.0, gamma='scale'
                      * For multi-class: Use decision_function_shape='ovr'
                    - Neural Networks: 
                      * MUST scale features
                      * Use MLPClassifier with hidden_layer_sizes=(100,50) for complex problems
                      * Set max_iter=500, early_stopping=True, validation_fraction=0.1
                      * Use solver='adam', learning_rate_init=0.001
                    
                    NOTE: Since preprocessing is handled separately, assume the dataset is clean and ready.
                    Focus on model training with appropriate hyperparameters for the problem type.
                    """,
                ),
                (
                    "human",
                    """Generate Python code for training and evaluating: {model_name}

Dataset metadata (read carefully - contains important characteristics):
{metadata_json}

Absolute dataset path: {dataset_path}
Problem type: {problem_type}
Target column: "{target_column}"
Test size: {test_size}

                    CRITICAL REQUIREMENTS - Follow these in order:

1. PATH AND LOADING:
   - Use os.path.abspath() for dataset: dataset_path = os.path.abspath(r"{dataset_path}")
   - Load dataset and print shape: print(f"Dataset shape: {{data.shape}}")
   - Target column is: "{target_column}"
   - Extract X (features) by dropping target column, extract y (target) column

2. DATA SPLITTING:
   - Split into train/test using train_test_split with random_state=42
   - Use stratify=y if classification problem with balanced classes
   - Set test_size={test_size}

3. FEATURE SCALING (Only if Required for Model):
   - For Logistic Regression, SVM, Neural Networks: 
     * MUST scale features with StandardScaler (fit on X_train, transform both)
   - For Random Forest, XGBoost, LightGBM: 
     * NO scaling needed (tree-based models don't require scaling)

4. MODEL TRAINING:
   - For Logistic Regression: 
     * MUST set max_iter=2000 or higher
     * For multi-class: Use multi_class='multinomial' or OneVsRestClassifier
     * Consider class_weight='balanced' if class imbalance detected
   - For Random Forest: 
     * Use n_estimators=200, max_depth=15, min_samples_split=5, random_state=42
     * Consider class_weight='balanced' for imbalanced classes
   - For XGBoost/LightGBM: 
     * Use appropriate objective for multi-class (e.g., 'multi:softprob')
     * Set learning_rate=0.1, n_estimators=200, max_depth=5-7
   - Set random_state=42 for reproducibility

5. EVALUATION:
   - Print metrics EXACTLY as shown (colon with space after):
     print("Accuracy:", score_value)
     print("Precision:", score_value)  # For multi-class: average='weighted'
     print("Recall:", score_value)     # For multi-class: average='weighted'
     print("F1-Score:", score_value)   # For multi-class: average='weighted'
     print("ROC-AUC:", score_value)    # For multi-class: multi_class='ovr', average='weighted'
   - If severe class imbalance: print per-class metrics or confusion matrix

IMPORTANT: The metadata shows these key characteristics:
- Problem type: {problem_type}
- Target column: "{target_column}"
- Target distribution: {target_distribution}
- Number of samples: {num_samples}
- Number of features: {num_features}
- Domain: {domain}

Use these characteristics to make smart modeling decisions:
- Choose appropriate hyperparameters based on dataset size and problem type
- For multi-class problems: use average='weighted' for metrics
- For class imbalance: consider class_weight='balanced'
- Select appropriate solver/objective for the model type

NOTE: Assume the dataset is already preprocessed and ready for modeling. 
Focus only on loading, splitting, scaling (if needed), training, and evaluation.

Generate complete, working Python code for model training and evaluation.
""",
                ),
            ]
        )
        self.chain: RunnableSequence = self.prompt | self.llm | StrOutputParser()
        
        # Error fixing prompt (concise and focused)
        self.fix_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert Python debugger. Fix errors in ML training code.
                    
                    Rules:
                    - Provide ONLY corrected Python code (no explanations)
                    - Fix the specific error while keeping working parts unchanged
                    - Ensure code is complete, runnable, and handles all data types correctly
                    - Common issues: date parsing, type conversions, column names, imports
                    - If dataset has date/datetime columns, convert them properly before modeling
                    - Ensure numeric operations only on numeric columns""",
                ),
                (
                    "human",
                    """Code that failed:
```python
{original_code}
```

Error: {error_message}

Return the complete fixed code.""",
                ),
            ]
        )
        self.fix_chain: RunnableSequence = self.fix_prompt | self.llm | StrOutputParser()
    
    def fix_code(self, original_code: str, error_message: str) -> str:
        """Fix code based on error message.
        
        Args:
            original_code: The code that failed
            error_message: The error message from execution
            
        Returns:
            Fixed code as a string
        """
        payload = {
            "original_code": original_code,
            "error_message": error_message,
        }
        
        fixed_code = self.fix_chain.invoke(payload)
        
        # Extract code block if wrapped in markdown
        fixed_code = self._extract_code_block(fixed_code)
        
        return fixed_code
    
    def generate_code(
        self,
        *,
        metadata: DatasetMetadata,
        dataset_path: Path,
        model_name: str,
        test_size: float = 0.2,
    ) -> str:
        """Generate Python code for training and validating a model.
        
        Args:
            metadata: Dataset metadata
            dataset_path: Path to the dataset file
            model_name: Name of the model to train (e.g., "Random Forest", "XGBoost")
            test_size: Fraction of data for testing
            
        Returns:
            Generated Python code as a string
        """
        # Extract key metadata characteristics for better context
        feature_types_summary = ", ".join([
            f"{col}: {dtype}" for col, dtype in list(metadata.feature_types.items())[:10]
        ])
        if len(metadata.feature_types) > 10:
            feature_types_summary += f" ... and {len(metadata.feature_types) - 10} more features"
        
        time_index_info = metadata.time_index if metadata.time_index else "None (not time-series)"
        target_dist = metadata.target_distribution or "Not specified in metadata"
        num_samples_str = f"{metadata.num_samples:,}" if metadata.num_samples else "Unknown"
        num_features_str = f"{metadata.num_features}" if metadata.num_features else "Unknown"
        domain_str = metadata.domain or "Unknown"
        
        payload = {
            "metadata_json": metadata.model_dump_json(indent=2),
            "dataset_path": str(dataset_path.absolute()),
            "model_name": model_name,
            "problem_type": metadata.problem_type,
            "target_column": metadata.target,
            "test_size": test_size,
            "feature_types_summary": feature_types_summary,
            "target_distribution": target_dist,
            "num_samples": num_samples_str,
            "num_features": num_features_str,
            "domain": domain_str,
            "time_index_info": time_index_info,
        }
        
        code = self.chain.invoke(payload)
        
        # Extract code block if wrapped in markdown
        code = self._extract_code_block(code)
        
        return code
    
    def _extract_code_block(self, text: str) -> str:
        """Extract Python code from markdown code block if present."""
        # Look for ```python or ``` code blocks
        pattern = r"```(?:python)?\n?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code block found, return as-is
        return text.strip()
    
    def save_code(self, code: str, output_path: Path, model_name: str) -> Path:
        """Save generated code to a file.
        
        Args:
            code: Generated Python code
            output_path: Directory to save the code file
            model_name: Model name for filename
            
        Returns:
            Path to saved file
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sanitize model name for filename
        safe_model_name = re.sub(r"[^\w\s-]", "", model_name.lower())
        safe_model_name = re.sub(r"[-\s]+", "_", safe_model_name)
        
        file_path = output_path / f"model_{safe_model_name}.py"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        return file_path
    
    def generate_full_training_code(
        self,
        *,
        metadata: DatasetMetadata,
        dataset_path: Path,
        model_name: str,
    ) -> str:
        """Generate Python code for full dataset training (no train/test split) and model saving.
        
        Args:
            metadata: Dataset metadata
            dataset_path: Path to the dataset file
            model_name: Name of the model to train (e.g., "Random Forest", "XGBoost")
            
        Returns:
            Generated Python code as a string
        """
        # Extract key metadata characteristics for better context
        target_dist = metadata.target_distribution or "Not specified in metadata"
        num_samples_str = f"{metadata.num_samples:,}" if metadata.num_samples else "Unknown"
        num_features_str = f"{metadata.num_features}" if metadata.num_features else "Unknown"
        domain_str = metadata.domain or "Unknown"
        
        # Create prompt for full training (no split, save model)
        full_training_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert ML engineer. Generate complete, standalone Python code 
                    for training a machine learning model on the FULL dataset (no train/test split)
                    and saving it as a .pkl file for production use (REST API integration).
                    
                    CRITICAL REQUIREMENTS:
                    
                    1. LOAD FULL DATASET:
                       - Load dataset using pandas from the provided absolute path
                       - Extract features (X) and target (y) based on target column name
                       - Print dataset shape for verification
                    
                    2. FEATURE SCALING (Only if Required):
                       - For Logistic Regression, SVM, Neural Networks: MUST scale with StandardScaler
                       - For tree-based models (Random Forest, XGBoost, LightGBM): NO scaling needed
                       - IMPORTANT: Fit scaler on ALL data (no train/test split)
                    
                    3. MODEL TRAINING:
                       - Train model on ENTIRE dataset (use all data for training)
                       - Use same hyperparameters as validation code
                       - Set random_state=42 for reproducibility
                       - For Logistic Regression: max_iter=2000+
                       - For multi-class: Use appropriate parameters (multi_class='multinomial', etc.)
                       - Consider class_weight='balanced' for imbalanced data
                    
                    4. SAVE MODEL:
                       - Save trained model using pickle: import pickle
                       - Model filename: model_{{model_name_safe}}.pkl
                       - Also save scaler if used: scaler_{{model_name_safe}}.pkl
                       - Use absolute paths for saving
                       - Print confirmation messages
                    
                    5. REST API READINESS:
                       - Code should produce model files ready for REST API integration
                       - Model should be saved in a way that can be loaded later for prediction
                       - Include example of how to load and use the model (as comments)
                    
                    CODE STRUCTURE:
                    - Include ALL imports at the top (pandas, sklearn, pickle, os)
                    - Use absolute paths for dataset and output files
                    - Wrap main code in try-except for error handling
                    - Print informative messages at each step
                    - Code must be production-ready and well-documented
                    
                    API COMPATIBILITY (scikit-learn 1.3+):
                    - LogisticRegression: max_iter=2000+, use appropriate solver
                    - For multi-class ROC-AUC: multi_class='ovr', average='weighted'
                    - Use StandardScaler for models requiring scaling
                    
                    The code should train on the FULL dataset and save the model for production use.
                    """,
                ),
                (
                    "human",
                    """Generate Python code for FULL DATASET training and model saving: {model_name}

Dataset metadata:
{metadata_json}

Absolute dataset path: {dataset_path}
Problem type: {problem_type}
Target column: "{target_column}"

CRITICAL REQUIREMENTS:
1. Load dataset from absolute path: {dataset_path}
2. Extract features (X) and target (y) from target column: "{target_column}"
3. Apply feature scaling ONLY if required for the model type
4. Train model on ENTIRE dataset (no splitting)
5. Save model as: model_{model_name_safe}.pkl using pickle
6. Save scaler as: scaler_{model_name_safe}.pkl if scaling was used
7. Print confirmation messages at each step

Key characteristics:
- Problem type: {problem_type}
- Target distribution: {target_distribution}
- Number of samples: {num_samples}
- Number of features: {num_features}
- Domain: {domain}

Use appropriate hyperparameters for {problem_type} problem.
Generate complete, production-ready code that trains on full dataset and saves the model.
The model filename should be: model_{model_name_safe}.pkl
The scaler filename should be: scaler_{model_name_safe}.pkl (if scaling is used)
""",
                ),
            ]
        )
        
        full_training_chain = full_training_prompt | self.llm | StrOutputParser()
        
        # Sanitize model name for filename
        safe_model_name = re.sub(r"[^\w\s-]", "", model_name.lower())
        safe_model_name = re.sub(r"[-\s]+", "_", safe_model_name)
        
        payload = {
            "metadata_json": metadata.model_dump_json(indent=2),
            "dataset_path": str(dataset_path.absolute()),
            "model_name": model_name,
            "model_name_safe": safe_model_name,
            "problem_type": metadata.problem_type,
            "target_column": metadata.target,
            "target_distribution": target_dist,
            "num_samples": num_samples_str,
            "num_features": num_features_str,
            "domain": domain_str,
        }
        
        code = full_training_chain.invoke(payload)
        
        # Extract code block if wrapped in markdown
        code = self._extract_code_block(code)
        
        return code
    
    def generate_validation_code(
        self,
        *,
        metadata: DatasetMetadata,
        dataset_path: Path,
        selected_models: List[str],
        test_size: float = 0.2,
    ) -> str:
        """Generate validation code with 80-20 split for all selected models.
        
        Args:
            metadata: Dataset metadata
            dataset_path: Path to the dataset file
            selected_models: List of model names to validate (e.g., ["Random Forest", "XGBoost"])
            test_size: Fraction of data for testing (default: 0.2 for 80-20 split)
            
        Returns:
            Generated Python code as a string
        """
        # Get model configurations
        model_configs = []
        for model_name in selected_models:
            config = self._get_model_config(model_name, metadata.problem_type)
            if config:
                model_configs.append({
                    "name": model_name,
                    "library": config["library"],
                    "class_name": config["class_name"],
                    "hyperparameters": config.get("hyperparameters", {}),
                    "needs_scaling": config.get("needs_scaling", False),
                })
        
        # Build code
        code = '''"""
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
'''
        
        # Add imports based on models
        sklearn_imports = set()
        xgb_import = False
        lgb_import = False
        
        for config in model_configs:
            if config["library"] == "sklearn":
                if config["needs_scaling"]:
                    sklearn_imports.add("preprocessing")
                sklearn_imports.add(self._get_sklearn_module(config["class_name"]))
            elif config["library"] == "xgboost":
                xgb_import = True
            elif config["library"] == "lightgbm":
                lgb_import = True
        
        # Add sklearn imports - import classes directly
        imported_classes = []
        for config in model_configs:
            if config["library"] == "sklearn":
                module_name = self._get_sklearn_module(config["class_name"])
                class_name = config["class_name"]
                if (module_name, class_name) not in imported_classes:
                    code += f"from sklearn.{module_name} import {class_name}\n"
                    imported_classes.append((module_name, class_name))
        # Always import StandardScaler
        code += "from sklearn.preprocessing import StandardScaler\n"
        
        # Add xgboost/lightgbm imports
        if xgb_import:
            code += "import xgboost as xgb\n"
        if lgb_import:
            code += "import lightgbm as lgb\n"
        
        # Determine metrics imports based on problem type
        if metadata.problem_type == "classification":
            metrics_imports_code = """from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)"""
        else:
            metrics_imports_code = """from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)"""
        
        code += f'''
{metrics_imports_code}
from sklearn.model_selection import train_test_split


def load_and_prepare_data(data_path: str, target_column: str):
    """Load dataset and prepare for training."""
    # Load data
    data_path = os.path.abspath(data_path)
    print(f"Loading data from {{data_path}}...")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {{data_path}}")
    
    print(f"Dataset shape: {{df.shape}}")
    
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
        print(f"Encoding {{len(categorical_cols)}} categorical features...")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    print(f"Final feature shape: {{X.shape}}")
    print(f"Target distribution: {{y.value_counts().to_dict()}}")
    
    return X, y


def validate_model(model_name: str, model_class, hyperparameters: dict, X, y, needs_scaling: bool, test_size: float = 0.2, problem_type: str = "{metadata.problem_type}"):
    """Validate model with 80-20 train/test split and return metrics."""
    print(f"\\n{'='*70}")
    print(f"VALIDATION TESTING - {{model_name}} (80-20 Split)")
    print(f"{'='*70}")
    
    # Split data (80-20)
    print(f"Splitting data: {{100*(1-test_size):.0f}}% train, {{test_size*100:.0f}}% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if problem_type == "classification" else None
    )
    print(f"Training samples: {{len(X_train)}}, Test samples: {{len(X_test)}}")
    
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
    print(f"Training {{model_name}} on training set...")
    model = model_class(**hyperparameters)
    model.fit(X_train_scaled, y_train_encoded)
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    metrics = {{"model_name": model_name}}
    
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
        
        print(f"\\nVALIDATION METRICS - {{model_name}}:")
        print(f"  Accuracy: {{metrics['accuracy']:.4f}}")
        print(f"  Precision (weighted): {{metrics['precision']:.4f}}")
        print(f"  Recall (weighted): {{metrics['recall']:.4f}}")
        print(f"  F1-Score (weighted): {{metrics['f1_score']:.4f}}")
        if metrics.get('roc_auc') is not None:
            print(f"  ROC-AUC: {{metrics['roc_auc']:.4f}}")
    
    else:  # regression
        metrics["mse"] = float(mean_squared_error(y_test_encoded, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test_encoded, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_test_encoded, y_pred))
        metrics["r2"] = float(r2_score(y_test_encoded, y_pred))
        
        print(f"\\nVALIDATION METRICS - {{model_name}}:")
        print(f"  MSE: {{metrics['mse']:.4f}}")
        print(f"  RMSE: {{metrics['rmse']:.4f}}")
        print(f"  MAE: {{metrics['mae']:.4f}}")
        print(f"  R²: {{metrics['r2']:.4f}}")
    
    return metrics


def main():
    """Main execution function."""
    try:
        # Configuration
        data_path = r"{str(dataset_path.absolute())}"
        target_column = "{metadata.target}"
        test_size = {test_size}  # 80-20 split for validation
        problem_type = "{metadata.problem_type}"
        
        print("="*70)
        print("MODEL VALIDATION TESTING (80-20 Split)")
        print("="*70)
        
        # Load and prepare data
        X, y = load_and_prepare_data(data_path, target_column)
        
        # Validate all models
        print("\\n" + "="*70)
        print("VALIDATION TESTING - All Models")
        print("="*70)
        validation_results = []
'''
        
        # Add validation calls for each model
        for config in model_configs:
            model_name = config["name"]
            library = config["library"]
            class_name = config["class_name"]
            hyperparams = config["hyperparameters"]
            needs_scaling = config["needs_scaling"]
            
            # Format hyperparameters as dictionary literal
            if hyperparams:
                params_list = []
                for k, v in hyperparams.items():
                    if isinstance(v, str):
                        params_list.append(f'"{k}": "{v}"')
                    else:
                        params_list.append(f'"{k}": {v}')
                params_str = ", ".join(params_list)
            else:
                params_str = ""
            
            # Get model class reference
            if library == "sklearn":
                model_class_ref = class_name
            elif library == "xgboost":
                model_class_ref = f"xgb.{class_name}"
            elif library == "lightgbm":
                model_class_ref = f"lgb.{class_name}"
            else:
                model_class_ref = class_name
            
            safe_model_name = model_name.lower().replace(' ', '_').replace('-', '_')
            
            code += f'''
        # Validate {model_name} with 80-20 split
        metrics_{safe_model_name} = validate_model(
            model_name="{model_name}",
            model_class={model_class_ref},
            hyperparameters={{{params_str}}},
            X=X,
            y=y,
            needs_scaling={needs_scaling},
            test_size=test_size,
            problem_type=problem_type,
        )
        validation_results.append(metrics_{safe_model_name})
'''
        
        code += '''
        # Display validation results summary
        print("\\n" + "="*70)
        print("VALIDATION TESTING RESULTS SUMMARY (80-20 Split)")
        print("="*70)
        for metrics in validation_results:
            model_name = metrics["model_name"]
            print(f"\\n{model_name}:")
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
        print(f"\\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
'''
        
        return code
    
    def generate_deployment_code(
        self,
        *,
        metadata: DatasetMetadata,
        dataset_path: Path,
        model_name: str,
    ) -> str:
        """Generate deployment code that trains a single model on full dataset and saves it as PKL file.
        
        Args:
            metadata: Dataset metadata
            dataset_path: Path to the dataset file
            model_name: Name of the best performing model to deploy (e.g., "Random Forest")
            
        Returns:
            Generated Python code as a string
        """
        # Get model configuration for the best model
        config = self._get_model_config(model_name, metadata.problem_type)
        if not config:
            raise ValueError(f"Could not find configuration for model: {model_name}")
        
        model_config = {
            "name": model_name,
            "library": config["library"],
            "class_name": config["class_name"],
            "hyperparameters": config.get("hyperparameters", {}),
            "needs_scaling": config.get("needs_scaling", False),
        }
        
        # Build code
        library = model_config["library"]
        class_name = model_config["class_name"]
        needs_scaling = model_config["needs_scaling"]
        hyperparams = model_config["hyperparameters"]
        
        code = '''"""
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
'''
        
        # Add imports based on model
        if library == "sklearn":
            module_name = self._get_sklearn_module(class_name)
            code += f"from sklearn.{module_name} import {class_name}\n"
        elif library == "xgboost":
            code += "import xgboost as xgb\n"
        elif library == "lightgbm":
            code += "import lightgbm as lgb\n"
        
        # Always import StandardScaler since it's used in train_and_save_model function
        code += "from sklearn.preprocessing import StandardScaler\n"
        
        # Format hyperparameters
        if hyperparams:
            params_list = []
            for k, v in hyperparams.items():
                if isinstance(v, str):
                    params_list.append(f'"{k}": "{v}"')
                else:
                    params_list.append(f'"{k}": {v}')
            params_str = ", ".join(params_list)
        else:
            params_str = ""
        
        # Get model class reference
        if library == "sklearn":
            model_class_ref = class_name
        elif library == "xgboost":
            model_class_ref = f"xgb.{class_name}"
        elif library == "lightgbm":
            model_class_ref = f"lgb.{class_name}"
        else:
            model_class_ref = class_name
        
        # Calculate safe_name at code generation time
        safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
        
        code += f'''

def load_and_prepare_data(data_path: str, target_column: str):
    """Load dataset and prepare for training."""
    # Load data
    data_path = os.path.abspath(data_path)
    print(f"Loading data from {{data_path}}...")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {{data_path}}")
    
    print(f"Dataset shape: {{df.shape}}")
    
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
        print(f"Encoding {{len(categorical_cols)}} categorical features...")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    
    print(f"Final feature shape: {{X.shape}}")
    print(f"Target distribution: {{y.value_counts().to_dict()}}")
    
    return X, y


def train_and_save_model(model_name: str, model_class, hyperparameters: dict, X, y, needs_scaling: bool, output_dir: str):
    """Train a single model on full dataset and save as PKL."""
    print(f"\\n{'='*70}")
    print(f"DEPLOYMENT TRAINING - {{model_name}} (Full Dataset)")
    print(f"{'='*70}")
    
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
    print(f"Training {{model_name}} on full dataset...")
    model = model_class(**hyperparameters)
    model.fit(X_scaled, y_encoded)
    print(f"[OK] {{model_name}} trained successfully")
    
    # Save model
    safe_model_name = model_name.lower().replace(' ', '_').replace('-', '_')
    model_filename = f"{{output_dir}}/model_{{safe_model_name}}.pkl"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving model to {{model_filename}}...")
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"[OK] Model saved: {{model_filename}}")
    
    # Save scaler if used
    if scaler is not None:
        safe_model_name = model_name.lower().replace(' ', '_').replace('-', '_')
        scaler_filename = f"{{output_dir}}/scaler_{{safe_model_name}}.pkl"
        print(f"Saving scaler to {{scaler_filename}}...")
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"[OK] Scaler saved: {{scaler_filename}}")
    
    # Save label encoder if used (for XGBoost/LightGBM)
    if label_encoder is not None:
        safe_model_name = model_name.lower().replace(' ', '_').replace('-', '_')
        encoder_filename = f"{{output_dir}}/label_encoder_{{safe_model_name}}.pkl"
        print(f"Saving label encoder to {{encoder_filename}}...")
        with open(encoder_filename, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"[OK] Label encoder saved: {{encoder_filename}}")
    
    return model, scaler


def main():
    """Main execution function."""
    try:
        # Configuration
        data_path = r"{str(dataset_path.absolute())}"
        target_column = "{metadata.target}"
        output_dir = "models"  # Directory to save PKL files
        
        print("="*70)
        print("DEPLOYMENT TRAINING - Full Dataset Training")
        print("="*70)
        
        # Load and prepare data
        X, y = load_and_prepare_data(data_path, target_column)
        
        # Train best performing model on full dataset and save PKL
        model_{safe_name}, scaler_{safe_name} = train_and_save_model(
            model_name="{model_name}",
            model_class={model_class_ref},
            hyperparameters={{{params_str}}},
            X=X,
            y=y,
            needs_scaling={needs_scaling},
            output_dir=output_dir,
        )
        
        print("\\n" + "="*70)
        print("DEPLOYMENT TRAINING COMPLETE")
        print("="*70)
        print(f"Trained {{'{model_name}'}} on full dataset")
        print(f"Model saved in: {{output_dir}}/")
        safe_model_name = "{model_name}".lower().replace(' ', '_').replace('-', '_')
        print("\\nSaved files:")
        print(f"  - model_{{safe_model_name}}.pkl")
        if {needs_scaling}:
            print(f"  - scaler_{{safe_model_name}}.pkl")
        model_name_lower = "{model_name}".lower()
        if 'xgboost' in model_name_lower or 'lightgbm' in model_name_lower:
            print(f"  - label_encoder_{{safe_model_name}}.pkl")
        
    except Exception as exc:
        print(f"\\nERROR: {{exc}}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
'''
        
        return code
    
    def _get_model_config(self, model_name: str, problem_type: str) -> dict:
        """Get model configuration from model name."""
        model_name_lower = model_name.lower()
        
        # Model mappings
        model_mappings = {
            "classification": {
                "logistic regression": {
                    "library": "sklearn",
                    "class_name": "LogisticRegression",
                    "hyperparameters": {"max_iter": 2000, "random_state": 42},
                    "needs_scaling": True,
                },
                "random forest": {
                    "library": "sklearn",
                    "class_name": "RandomForestClassifier",
                    "hyperparameters": {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "random_state": 42},
                    "needs_scaling": False,
                },
                "decision tree": {
                    "library": "sklearn",
                    "class_name": "DecisionTreeClassifier",
                    "hyperparameters": {"random_state": 42},
                    "needs_scaling": False,
                },
                "svm": {
                    "library": "sklearn",
                    "class_name": "SVC",
                    "hyperparameters": {"probability": True, "random_state": 42},
                    "needs_scaling": True,
                },
                "gradient boosting": {
                    "library": "sklearn",
                    "class_name": "GradientBoostingClassifier",
                    "hyperparameters": {"random_state": 42},
                    "needs_scaling": False,
                },
                "xgboost": {
                    "library": "xgboost",
                    "class_name": "XGBClassifier",
                    "hyperparameters": {"random_state": 42},
                    "needs_scaling": False,
                },
                "lightgbm": {
                    "library": "lightgbm",
                    "class_name": "LGBMClassifier",
                    "hyperparameters": {"random_state": 42},
                    "needs_scaling": False,
                },
            },
            "regression": {
                "linear regression": {
                    "library": "sklearn",
                    "class_name": "LinearRegression",
                    "hyperparameters": {},
                    "needs_scaling": True,
                },
                "random forest": {
                    "library": "sklearn",
                    "class_name": "RandomForestRegressor",
                    "hyperparameters": {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "random_state": 42},
                    "needs_scaling": False,
                },
                "decision tree": {
                    "library": "sklearn",
                    "class_name": "DecisionTreeRegressor",
                    "hyperparameters": {"random_state": 42},
                    "needs_scaling": False,
                },
                "svm": {
                    "library": "sklearn",
                    "class_name": "SVR",
                    "hyperparameters": {},
                    "needs_scaling": True,
                },
                "gradient boosting": {
                    "library": "sklearn",
                    "class_name": "GradientBoostingRegressor",
                    "hyperparameters": {"random_state": 42},
                    "needs_scaling": False,
                },
                "xgboost": {
                    "library": "xgboost",
                    "class_name": "XGBRegressor",
                    "hyperparameters": {"random_state": 42},
                    "needs_scaling": False,
                },
                "lightgbm": {
                    "library": "lightgbm",
                    "class_name": "LGBMRegressor",
                    "hyperparameters": {"random_state": 42},
                    "needs_scaling": False,
                },
            },
        }
        
        if problem_type not in model_mappings:
            return None
        
        for key, config in model_mappings[problem_type].items():
            if key in model_name_lower or model_name_lower in key:
                return config
        
        return None
    
    def _get_sklearn_module(self, class_name: str) -> str:
        """Get sklearn module name for a class."""
        if "LogisticRegression" in class_name or "LinearRegression" in class_name or "Ridge" in class_name:
            return "linear_model"
        elif "RandomForest" in class_name or "GradientBoosting" in class_name:
            return "ensemble"
        elif "DecisionTree" in class_name:
            return "tree"
        elif "SVC" in class_name or "SVR" in class_name:
            return "svm"
        elif "KNeighbors" in class_name:
            return "neighbors"
        else:
            return "ensemble"
