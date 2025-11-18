"""Main CLI entry point for the ML model recommendation agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from src.metadata_generator import generate_metadata_from_file
from src.model_recommender import DatasetMetadata, ModelRecommenderAgent
from src.pipeline_executor import PipelineExecutor

app = typer.Typer(help="ML Model Recommender Agent - Get model recommendations and generate validation code")
console = Console()


@app.command()
def generate_metadata(
    dataset_path: Path = typer.Argument(..., help="Path to dataset file (CSV, Parquet, or JSON)"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output path for metadata JSON"),
    name: Optional[str] = typer.Option(None, "--name", help="Custom dataset name"),
    target: Optional[str] = typer.Option(None, "--target", help="Target column name (auto-detected if not specified)"),
    problem_type: Optional[str] = typer.Option(None, "--problem-type", help="Problem type (classification, regression, etc.)"),
    domain: Optional[str] = typer.Option(None, "--domain", help="Domain context (e.g., finance, telecom)"),
    constraints: Optional[str] = typer.Option(None, "--constraints", help="Comma-separated list of constraints"),
    metric: Optional[str] = typer.Option(None, "--metric", help="Evaluation metric (e.g., accuracy, rmse)"),
    notes: Optional[str] = typer.Option(None, "--notes", help="Additional notes about the dataset"),
):
    """Generate dataset metadata JSON from a data file."""
    console.print(f"\n[bold blue]Analyzing dataset: {dataset_path}[/bold blue]")
    
    # Parse constraints
    constraints_list = None
    if constraints:
        constraints_list = [c.strip() for c in constraints.split(",")]
    
    try:
        metadata, output_path = generate_metadata_from_file(
            data_path=dataset_path,
            output_path=output,
            name=name,
            target=target,
            problem_type=problem_type,
            domain=domain,
            constraints=constraints_list,
            evaluation_metric=metric,
            notes=notes,
        )
        
        console.print(f"\n[bold green]✓ Metadata generated successfully![/bold green]")
        console.print(f"\nSaved to: {output_path}")
        console.print(f"\n[bold]Metadata Summary:[/bold]")
        console.print(f"  Name: {metadata.name}")
        console.print(f"  Problem type: {metadata.problem_type}")
        console.print(f"  Target: {metadata.target}")
        console.print(f"  Samples: {metadata.num_samples:,}")
        console.print(f"  Features: {metadata.num_features}")
        console.print(f"  Domain: {metadata.domain or 'N/A'}")
        console.print(f"  Time index: {metadata.time_index or 'N/A'}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def recommend(
    metadata_path: Path = typer.Argument(..., help="Path to metadata JSON file"),
    context: Optional[str] = typer.Option(None, "-c", "--context", help="Extra context for recommendations"),
    context_file: Optional[Path] = typer.Option(None, "--context-file", help="Path to file containing extra context"),
):
    """Get model recommendations from LLM based on dataset metadata."""
    console.print(f"\n[bold blue]Loading metadata from: {metadata_path}[/bold blue]")
    
    # Load metadata
    try:
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)
        metadata = DatasetMetadata(**metadata_dict)
    except Exception as e:
        console.print(f"[bold red]Error loading metadata:[/bold red] {e}")
        raise typer.Exit(1)
    
    # Load context from file if provided
    if context_file:
        try:
            with open(context_file, "r") as f:
                context = f.read()
        except Exception as e:
            console.print(f"[bold red]Error loading context file:[/bold red] {e}")
            raise typer.Exit(1)
    
    # Get recommendations
    console.print("\n[bold blue]Getting model recommendations from LLM...[/bold blue]")
    agent = ModelRecommenderAgent()
    recommendations = agent.recommend(metadata=metadata, context=context)
    
    # Display recommendations
    console.print("\n[bold green]Model Recommendations:[/bold green]")
    console.print(Markdown(recommendations))


@app.command("train-models")
def train_models(
    metadata_path: Path = typer.Argument(..., help="Path to metadata JSON file"),
    dataset_path: Path = typer.Argument(..., help="Path to dataset file (CSV, Parquet, or JSON)"),
    context: Optional[str] = typer.Option(None, "-c", "--context", help="Extra context for recommendations"),
    context_file: Optional[Path] = typer.Option(None, "--context-file", help="Path to file containing extra context"),
    test_size: float = typer.Option(0.2, "--test-size", help="Fraction of data for testing (default: 0.2)"),
    min_models: int = typer.Option(1, "--min-models", help="Minimum number of models to select (default: 1)"),
    max_models: int = typer.Option(3, "--max-models", help="Maximum number of models to select (default: 3)"),
    code_dir: Path = typer.Option(Path("generated_code"), "--code-dir", help="Directory to save generated code (default: generated_code)"),
):
    """Complete pipeline: recommendations -> model selection -> code generation -> training & evaluation."""
    console.print(f"\n[bold blue]Starting complete pipeline...[/bold blue]")
    console.print(f"  Metadata: {metadata_path}")
    console.print(f"  Dataset: {dataset_path}")
    
    # Load metadata
    try:
        with open(metadata_path, "r") as f:
            metadata_dict = json.load(f)
        metadata = DatasetMetadata(**metadata_dict)
    except Exception as e:
        console.print(f"[bold red]Error loading metadata:[/bold red] {e}")
        raise typer.Exit(1)
    
    # Load context from file if provided
    if context_file:
        try:
            with open(context_file, "r") as f:
                context = f.read()
        except Exception as e:
            console.print(f"[bold red]Error loading context file:[/bold red] {e}")
            raise typer.Exit(1)
    
    # Verify dataset exists
    if not dataset_path.exists():
        console.print(f"[bold red]Error: Dataset file not found: {dataset_path}[/bold red]")
        raise typer.Exit(1)
    
    # Execute pipeline
    try:
        executor = PipelineExecutor()
        results = executor.execute_pipeline(
            metadata=metadata,
            dataset_path=dataset_path,
            context=context,
            test_size=test_size,
            min_models=min_models,
            max_models=max_models,
            code_dir=code_dir,
        )
        
        if results:
            console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]")
        else:
            console.print("\n[yellow]Pipeline completed with no results.[/yellow]")
            
    except Exception as e:
        console.print(f"\n[bold red]Error executing pipeline:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

