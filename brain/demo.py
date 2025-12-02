"""
Demo script to test the Reasoning Agent Brain system.
"""

from pathlib import Path
from brain.memory import MemoryManager
from rich.console import Console

console = Console()


def demo_brain_system():
    """Demonstrate the brain system functionality."""
    console.print("[bold blue]Reasoning Agent Brain Demo[/bold blue]\n")
    
    # Initialize memory manager
    memory = MemoryManager()
    console.print("[green]✓ Memory Manager initialized[/green]")
    
    # Try to load report.json from project root
    project_root = Path(__file__).parent.parent.parent
    report_path = project_root / "report.json"
    
    if report_path.exists():
        console.print(f"[green]✓ Found report.json at {report_path}[/green]")
        
        # Load and store dataset analysis
        report_data = memory.load_report_json(str(report_path))
        if report_data:
            experiment_id = memory.store_dataset_analysis(
                dataset_path=str(project_root / "Datasets" / "Housing.csv"),
                target_column="price",
                report_data=report_data,
                report_path=str(report_path)
            )
            console.print(f"[green]✓ Stored dataset analysis (Experiment ID: {experiment_id})[/green]")
            
            # Store sample model parameters
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
            console.print("[green]✓ Stored model parameters[/green]")
            
            # Store sample evaluation results (before improvement)
            memory.store_evaluation_results(
                experiment_id=experiment_id,
                model_name="Random Forest",
                metrics={
                    "rmse": 0.5234,
                    "mae": 0.4123,
                    "r2_score": 0.8234
                },
                execution_info={
                    "status": "success",
                    "version": "original"
                }
            )
            console.print("[green]✓ Stored evaluation results (original)[/green]")
            
            # Store improved evaluation results (after improvement)
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
                    "version": "improved"
                }
            )
            console.print("[green]✓ Stored evaluation results (improved)[/green]")
            
            # Retrieve and display history
            console.print("\n[bold]Retrieving experiment history...[/bold]")
            history = memory.get_experiment_history(experiment_id=experiment_id)
            
            if history:
                console.print(f"[green]✓ Retrieved history for experiment {experiment_id}[/green]")
                console.print(f"  Dataset: {history['dataset_analysis']['dataset_path']}")
                console.print(f"  Target: {history['dataset_analysis']['target_column']}")
                console.print(f"  Models: {len(history['model_parameters']['models']) if history['model_parameters'] else 0}")
                console.print(f"  Evaluations: {len(history['evaluation_results']['evaluations']) if history['evaluation_results'] else 0}")
            
            # Compare before/after
            console.print("\n[bold]Comparing before/after improvements...[/bold]")
            comparison = memory.compare_before_after(experiment_id, "Random Forest")
            
            if comparison:
                console.print("[green]✓ Comparison generated[/green]")
                console.print(f"\n  Model: {comparison['model_name']}")
                console.print(f"  Before: {comparison['before']['timestamp'][:10]}")
                console.print(f"  After: {comparison['after']['timestamp'][:10]}")
                
                if comparison["improvements"]:
                    console.print("\n  Improvements:")
                    for metric, data in comparison["improvements"].items():
                        status = "✓" if data["improved"] else "✗"
                        console.print(f"    {status} {metric}: {data['improvement_pct']:+.2f}% "
                                    f"({data['before']:.4f} → {data['after']:.4f})")
            
            # List all experiments
            console.print("\n[bold]All stored experiments:[/bold]")
            all_experiments = memory.get_all_experiments()
            console.print(f"  Total: {len(all_experiments)}")
            for exp in all_experiments:
                console.print(f"    - {exp['experiment_id']}: {Path(exp['dataset_path']).name} ({exp['target_column']})")
            
            console.print("\n[bold green]✓ Demo completed successfully![/bold green]")
        else:
            console.print("[yellow]⚠ Could not load report.json data[/yellow]")
    else:
        console.print(f"[yellow]⚠ report.json not found at {report_path}[/yellow]")
        console.print("[yellow]  Creating sample data instead...[/yellow]")
        
        # Create sample data
        sample_report = {
            "dataset_overview": {
                "rows": 100,
                "columns": 5,
                "target_column": "target",
                "target_type": "numeric",
                "dataset_size_flag": "small"
            },
            "target_analysis": {
                "type": "numeric",
                "mean": 10.5,
                "std": 2.3,
                "skew": 0.5
            }
        }
        
        experiment_id = memory.store_dataset_analysis(
            dataset_path="sample_dataset.csv",
            target_column="target",
            report_data=sample_report
        )
        console.print(f"[green]✓ Created sample experiment (ID: {experiment_id})[/green]")


if __name__ == "__main__":
    demo_brain_system()

