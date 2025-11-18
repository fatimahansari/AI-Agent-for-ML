# Complete Agent Workflow Guide

This guide shows how to run the entire ML orchestration agent sequentially from start to finish.

## Workflow Overview

The agent has three main commands that can be run sequentially:

1. **`generate-metadata`** - Analyze dataset and create metadata JSON
2. **`recommend`** - Get model recommendations from LLM (optional)
3. **`train-models`** - Complete pipeline: recommendations → model selection → code generation → training & evaluation

## Option 1: Complete Automated Workflow (Recommended)

This runs everything in one command after generating metadata:

### Step 1: Generate Metadata
```powershell
# Basic - auto-detect everything
python -m src.main generate-metadata examples/AAPL.csv

# Advanced - with custom options
python -m src.main generate-metadata examples/AAPL.csv ^
  --output examples/AAPL_metadata.json ^
  --target "Close_forcast" ^
  --problem-type regression ^
  --domain finance ^
  --metric rmse ^
  --constraints "Temporal ordering must be preserved,Use time-based cross-validation" ^
  --notes "Stock price forecasting with technical indicators"
```

### Step 2: Train Models (Complete Pipeline)
```powershell
# This command does everything:
# - Gets model recommendations from LLM
# - Prompts you to select 1-3 models interactively
# - Generates Python code for selected models
# - Trains and evaluates models with train/test split
# - Displays evaluation metrics

python -m src.main train-models examples/AAPL_metadata.json examples/AAPL.csv ^
  --context "Need near-real-time scoring within a REST API" ^
  --test-size 0.2 ^
  --min-models 1 ^
  --max-models 3 ^
  --code-dir generated_code
```

**What happens in `train-models`:**
1. ✅ Loads metadata and dataset
2. ✅ Gets model recommendations from LLM
3. ✅ Displays recommendations and prompts for model selection (1-3 models)
4. ✅ Generates Python code files for selected models
5. ✅ Executes generated code to train models
6. ✅ Displays evaluation metrics (accuracy, RMSE, etc.)

## Option 2: Step-by-Step Workflow (For Exploration)

If you want to see recommendations first before training:

### Step 1: Generate Metadata
```powershell
python -m src.main generate-metadata examples/AAPL.csv ^
  --output examples/AAPL_metadata.json ^
  --target "Close_forcast" ^
  --problem-type regression
```

### Step 2: Get Recommendations Only (Optional)
```powershell
# View recommendations without training
python -m src.main recommend examples/AAPL_metadata.json ^
  --context "Need near-real-time scoring within a REST API"
```

### Step 3: Train Models
```powershell
# After reviewing recommendations, proceed with training
python -m src.main train-models examples/AAPL_metadata.json examples/AAPL.csv ^
  --context "Need near-real-time scoring within a REST API" ^
  --test-size 0.2
```

## Complete Example: AAPL Dataset

```powershell
# Step 1: Generate metadata
python -m src.main generate-metadata examples/AAPL.csv ^
  --output examples/AAPL_metadata.json ^
  --target "Close_forcast" ^
  --problem-type regression ^
  --domain finance ^
  --metric rmse

# Step 2: Train models (includes recommendations, selection, code generation, training)
python -m src.main train-models examples/AAPL_metadata.json examples/AAPL.csv ^
  --context "Stock price forecasting with low latency requirements" ^
  --test-size 0.2 ^
  --max-models 3
```

## Complete Example: Customer Churn Dataset

```powershell
# Step 1: Generate metadata (or use existing)
python -m src.main generate-metadata examples/customer_churn.csv ^
  --output examples/customer_churn_metadata.json ^
  --target "churned" ^
  --problem-type classification ^
  --domain telecom

# Step 2: Train models
python -m src.main train-models examples/customer_churn_metadata.json examples/customer_churn.csv ^
  --context "Need explainable predictions for business stakeholders, latency < 300ms" ^
  --test-size 0.2
```

## Command Reference

### `generate-metadata`
- **Purpose**: Analyze dataset and create metadata JSON
- **Required**: Dataset file path
- **Output**: Metadata JSON file
- **Auto-detects**: Target column, problem type, feature types, missing values, domain

### `recommend`
- **Purpose**: Get model recommendations from LLM
- **Required**: Metadata JSON file
- **Output**: Markdown list of recommended models with explanations
- **Note**: This is automatically done in `train-models`, so optional if using that command

### `train-models`
- **Purpose**: Complete end-to-end pipeline
- **Required**: Metadata JSON file + Dataset file
- **Output**: 
  - Model recommendations (displayed)
  - Generated Python code files (in `generated_code/` directory)
  - Trained models and evaluation metrics (displayed)
- **Interactive**: Prompts you to select 1-3 models from recommendations

## Options Reference

### Common Options
- `--context` / `-c`: Extra narrative context (constraints, deployment, etc.)
- `--context-file`: Path to file containing extra context

### `train-models` Specific Options
- `--test-size`: Fraction of data for testing (default: 0.2 = 80/20 split)
- `--min-models`: Minimum models to select (default: 1)
- `--max-models`: Maximum models to select (default: 3)
- `--code-dir`: Directory to save generated code (default: `generated_code`)

### `generate-metadata` Specific Options
- `--output` / `-o`: Output path for metadata JSON
- `--target`: Target column name (auto-detected if not specified)
- `--problem-type`: classification, regression, clustering, time-series, nlp, recommendation, anomaly-detection
- `--domain`: Domain context (e.g., finance, telecom, healthcare)
- `--metric`: Evaluation metric (e.g., accuracy, rmse, roc-auc)
- `--constraints`: Comma-separated list of constraints
- `--notes`: Additional notes about the dataset

## Output Files

After running `train-models`, you'll find:
- `generated_code/model_<model_name>.py` - Standalone Python files for each selected model
- These files can be run independently: `python generated_code/model_random_forest.py`

## Tips

1. **Start with metadata generation** - Always generate metadata first to ensure proper dataset analysis
2. **Use context** - Provide context about constraints, deployment, or requirements for better recommendations
3. **Review recommendations** - If using `train-models`, you'll see recommendations and can select which models to train
4. **Check generated code** - Review the generated Python files in `generated_code/` directory
5. **Reuse metadata** - Once metadata is generated, you can use it multiple times with different contexts

