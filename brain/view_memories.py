"""
Utility script to view stored memories in the Reasoning Agent Brain.
"""

import json
from pathlib import Path
from brain.memory import MemoryManager
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON

console = Console()


def view_all_experiments():
    """Display all stored experiments."""
    memory = MemoryManager()
    experiments = memory.get_all_experiments()
    
    if not experiments:
        console.print("[yellow]No experiments stored yet.[/yellow]")
        return
    
    table = Table(title="Stored Experiments")
    table.add_column("Experiment ID", style="cyan")
    table.add_column("Dataset", style="green")
    table.add_column("Target", style="magenta")
    table.add_column("Created", style="dim")
    table.add_column("Last Updated", style="dim")
    
    for exp in experiments:
        table.add_row(
            exp["experiment_id"],
            Path(exp["dataset_path"]).name,
            exp["target_column"],
            exp["created_at"][:10],  # Just date
            exp["last_updated"][:10]
        )
    
    console.print(table)


def view_experiment_details(experiment_id: str):
    """Display detailed information about a specific experiment."""
    memory = MemoryManager()
    history = memory.get_experiment_history(experiment_id=experiment_id)
    
    if not history:
        console.print(f"[red]Experiment {experiment_id} not found.[/red]")
        return
    
    # Dataset Analysis
    if history["dataset_analysis"]:
        analysis = history["dataset_analysis"]
        console.print(Panel.fit(
            f"[bold]Dataset Analysis[/bold]\n"
            f"Dataset: {analysis['dataset_path']}\n"
            f"Target: {analysis['target_column']}\n"
            f"Timestamp: {analysis['timestamp']}",
            title="Dataset Analysis"
        ))
        
        if "analysis" in analysis:
            overview = analysis["analysis"].get("dataset_overview", {})
            console.print(f"\n[bold]Overview:[/bold]")
            console.print(f"  Rows: {overview.get('rows', 'N/A')}")
            console.print(f"  Columns: {overview.get('columns', 'N/A')}")
            console.print(f"  Target Type: {overview.get('target_type', 'N/A')}")
            console.print(f"  Dataset Size: {overview.get('dataset_size_flag', 'N/A')}")
    
    # Model Parameters
    if history["model_parameters"]:
        models = history["model_parameters"].get("models", [])
        if models:
            console.print(f"\n[bold]Model Parameters ({len(models)} models):[/bold]")
            for model in models:
                console.print(f"\n  [cyan]{model['model_name']}[/cyan]")
                console.print(f"    Class: {model['model_class']}")
                console.print(f"    Problem Type: {model['problem_type']}")
                console.print(f"    Test Size: {model['test_size']}")
                console.print(f"    Hyperparameters: {json.dumps(model['hyperparameters'], indent=6)}")
    
    # Evaluation Results
    if history["evaluation_results"]:
        evaluations = history["evaluation_results"].get("evaluations", [])
        if evaluations:
            console.print(f"\n[bold]Evaluation Results ({len(evaluations)} evaluations):[/bold]")
            
            # Group by model
            models_dict = {}
            for eval_result in evaluations:
                model_name = eval_result["model_name"]
                if model_name not in models_dict:
                    models_dict[model_name] = []
                models_dict[model_name].append(eval_result)
            
            for model_name, evals in models_dict.items():
                console.print(f"\n  [cyan]{model_name}[/cyan] ({len(evals)} evaluations):")
                
                # Show latest evaluation
                latest = sorted(evals, key=lambda x: x["timestamp"], reverse=True)[0]
                console.print(f"    Latest ({latest['timestamp'][:10]}):")
                for metric, value in latest["metrics"].items():
                    console.print(f"      {metric}: {value:.4f}" if isinstance(value, (int, float)) else f"      {metric}: {value}")


def compare_model_improvements(experiment_id: str, model_name: str):
    """Compare before/after improvements for a model."""
    memory = MemoryManager()
    comparison = memory.compare_before_after(experiment_id, model_name)
    
    if not comparison:
        console.print(f"[yellow]No comparison available for {model_name} in experiment {experiment_id}[/yellow]")
        return
    
    console.print(Panel.fit(
        f"[bold]{model_name}[/bold]\n"
        f"Before: {comparison['before']['timestamp'][:10]}\n"
        f"After: {comparison['after']['timestamp'][:10]}",
        title="Model Comparison"
    ))
    
    if comparison["improvements"]:
        table = Table(title="Improvements")
        table.add_column("Metric", style="cyan")
        table.add_column("Before", style="yellow")
        table.add_column("After", style="green")
        table.add_column("Change %", style="magenta")
        table.add_column("Status", style="bold")
        
        for metric, data in comparison["improvements"].items():
            status = "✓ Improved" if data["improved"] else "✗ Degraded"
            status_style = "green" if data["improved"] else "red"
            
            table.add_row(
                metric,
                f"{data['before']:.4f}",
                f"{data['after']:.4f}",
                f"{data['improvement_pct']:+.2f}%",
                f"[{status_style}]{status}[/{status_style}]"
            )
        
        console.print(table)
    else:
        console.print("[yellow]No comparable metrics found.[/yellow]")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            view_all_experiments()
        elif sys.argv[1] == "view" and len(sys.argv) > 2:
            view_experiment_details(sys.argv[2])
        elif sys.argv[1] == "compare" and len(sys.argv) > 3:
            compare_model_improvements(sys.argv[2], sys.argv[3])
        else:
            console.print("[red]Usage:[/red]")
            console.print("  python view_memories.py list")
            console.print("  python view_memories.py view <experiment_id>")
            console.print("  python view_memories.py compare <experiment_id> <model_name>")
    else:
        view_all_experiments()

