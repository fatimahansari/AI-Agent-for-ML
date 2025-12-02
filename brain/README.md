# Reasoning Agent Brain

The Brain module provides persistent memory for the Reasoning Agent, storing:
- Dataset analysis (from report.json)
- Model parameters used
- Evaluation results generated

## Purpose

This allows the Reasoning Agent to:
1. **Remember** what datasets it has analyzed
2. **Track** which models and parameters were used
3. **Compare** evaluation results before and after improvements
4. **Learn** from past experiments

## Structure

```
brain/
├── __init__.py          # Module initialization
├── memory.py            # MemoryManager class
├── README.md            # This file
└── memories/            # Storage directory (created automatically)
    ├── index.json       # Index of all experiments
    └── {experiment_id}/ # Per-experiment storage
        ├── dataset_analysis.json
        ├── model_parameters.json
        └── evaluation_results.json
```

## Usage

### Basic Usage

```python
from brain.memory import MemoryManager

# Initialize memory manager
memory = MemoryManager()

# Store dataset analysis
report_data = memory.load_report_json("../report.json")
experiment_id = memory.store_dataset_analysis(
    dataset_path="Datasets/Housing.csv",
    target_column="price",
    report_data=report_data,
    report_path="../report.json"
)

# Store model parameters
memory.store_model_parameters(
    experiment_id=experiment_id,
    model_name="Random Forest",
    model_class="RandomForestRegressor",
    hyperparameters={
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "random_state": 42
    },
    problem_type="regression",
    test_size=0.2
)

# Store evaluation results
memory.store_evaluation_results(
    experiment_id=experiment_id,
    model_name="Random Forest",
    metrics={
        "rmse": 0.4523,
        "mae": 0.3214,
        "r2_score": 0.8765
    },
    execution_info={
        "status": "success",
        "timestamp": "2024-01-01T12:00:00"
    }
)
```

### Retrieving History

```python
# Get complete experiment history
history = memory.get_experiment_history(
    dataset_path="Datasets/Housing.csv",
    target_column="price"
)

# Get all experiments
all_experiments = memory.get_all_experiments()

# Compare before/after improvements
comparison = memory.compare_before_after(
    experiment_id=experiment_id,
    model_name="Random Forest"
)
```

## Integration with Pipeline

The MemoryManager can be integrated into:
- `PipelineExecutor` - to store results after execution
- `TestExecutor` - to store metrics after evaluation
- `ModelTrainer` - to store model parameters

## Experiment ID

Each unique dataset/target combination gets a unique experiment ID (12-character hash). This allows:
- Multiple experiments on the same dataset
- Tracking improvements over time
- Comparing different model configurations

## Storage Format

All data is stored as JSON for:
- Human readability
- Easy debugging
- Simple integration with other tools

## Limitations

- Maximum 10 evaluation results per model (to prevent file bloat)
- No automatic cleanup of old experiments
- Single-threaded access (no locking mechanism)

