"""
Example integration of MemoryManager with Reasoning Agent pipeline.

This shows how to integrate the brain system into the existing pipeline.
"""

from pathlib import Path
from brain.memory import MemoryManager
from src.model_recommender import DatasetMetadata


def integrate_with_pipeline_executor():
    """
    Example of how to integrate MemoryManager into PipelineExecutor.
    
    This would be added to the execute_pipeline method.
    """
    memory = MemoryManager()
    
    # After loading metadata and dataset
    # dataset_path = ...
    # metadata = ...
    
    # Load and store dataset analysis
    report_path = Path(__file__).parent.parent.parent / "report.json"
    if report_path.exists():
        report_data = memory.load_report_json(str(report_path))
        if report_data:
            experiment_id = memory.store_dataset_analysis(
                dataset_path=str(dataset_path),
                target_column=metadata.target,
                report_data=report_data,
                report_path=str(report_path)
            )
        else:
            experiment_id = None
    else:
        experiment_id = None
    
    # After model selection and code generation
    # selected_models = ...
    # model_configs = ...
    
    # Store model parameters for each selected model
    if experiment_id:
        for model_config in model_configs:
            memory.store_model_parameters(
                experiment_id=experiment_id,
                model_name=model_config["name"],
                model_class=model_config["class_name"],
                hyperparameters=model_config.get("hyperparameters", {}),
                problem_type=metadata.problem_type,
                test_size=0.2,
                additional_info={
                    "library": model_config.get("library", ""),
                    "needs_scaling": model_config.get("needs_scaling", False)
                }
            )
    
    # After execution and getting results
    # results = ...
    
    # Store evaluation results
    if experiment_id:
        for result in results:
            if result.get("status") == "success":
                memory.store_evaluation_results(
                    experiment_id=experiment_id,
                    model_name=result.get("model_name", "Unknown"),
                    metrics=result.get("metrics", {}),
                    execution_info={
                        "status": result.get("status"),
                        "return_code": result.get("return_code", 0),
                        "stdout_length": len(result.get("stdout", "")),
                        "stderr_length": len(result.get("stderr", ""))
                    }
                )
    
    return experiment_id


def compare_improvements_example():
    """Example of comparing before/after improvements."""
    memory = MemoryManager()
    
    # Get experiment ID
    experiment_id = memory._generate_experiment_id(
        "Datasets/Housing.csv",
        "price"
    )
    
    # Compare results
    comparison = memory.compare_before_after(
        experiment_id=experiment_id,
        model_name="Random Forest"
    )
    
    if comparison:
        print(f"Model: {comparison['model_name']}")
        print("\nImprovements:")
        for metric, data in comparison["improvements"].items():
            if data["improved"]:
                print(f"  {metric}: {data['improvement_pct']:.2f}% improvement")
                print(f"    Before: {data['before']:.4f}")
                print(f"    After: {data['after']:.4f}")


if __name__ == "__main__":
    # Example usage
    memory = MemoryManager()
    
    # Load report.json
    report_path = Path(__file__).parent.parent.parent / "report.json"
    if report_path.exists():
        report_data = memory.load_report_json(str(report_path))
        if report_data:
            experiment_id = memory.store_dataset_analysis(
                dataset_path="Datasets/Housing.csv",
                target_column="price",
                report_data=report_data,
                report_path=str(report_path)
            )
            print(f"Stored dataset analysis with experiment_id: {experiment_id}")
            
            # Get all experiments
            experiments = memory.get_all_experiments()
            print(f"\nTotal experiments stored: {len(experiments)}")
            
            # Get experiment history
            history = memory.get_experiment_history(experiment_id=experiment_id)
            if history:
                print(f"\nExperiment history retrieved:")
                print(f"  Dataset: {history['dataset_analysis']['dataset_path']}")
                print(f"  Target: {history['dataset_analysis']['target_column']}")

