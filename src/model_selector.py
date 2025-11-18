from __future__ import annotations

from typing import List

import typer

from src.model_recommender import ModelRecommendation


def select_models(recommendations: List[ModelRecommendation], min_selection: int = 1, max_selection: int = 3) -> List[ModelRecommendation]:
    """Interactive console interface to select models from recommendations."""
    typer.echo("\n" + "=" * 70)
    typer.echo("MODEL RECOMMENDATIONS")
    typer.echo("=" * 70 + "\n")
    
    for idx, rec in enumerate(recommendations, 1):
        typer.echo(f"{idx}. {rec.name}")
        typer.echo(f"   Library: {rec.library}.{rec.class_name}")
        typer.echo(f"   Reason: {rec.reason}")
        typer.echo(f"   Preprocessing: {rec.preprocessing}")
        if rec.hyperparameters:
            typer.echo(f"   Suggested hyperparameters: {rec.hyperparameters}")
        typer.echo()
    
    typer.echo(f"\nPlease select {min_selection} to {max_selection} models (comma-separated numbers, e.g., 1,2,3):")
    
    while True:
        try:
            selection = typer.prompt("Your selection", default="1")
            indices = [int(x.strip()) - 1 for x in selection.split(",")]
            
            # Validate selection
            if len(indices) < min_selection:
                typer.echo(f"❌ Please select at least {min_selection} model(s).", err=True)
                continue
            if len(indices) > max_selection:
                typer.echo(f"❌ Please select at most {max_selection} model(s).", err=True)
                continue
            if any(idx < 0 or idx >= len(recommendations) for idx in indices):
                typer.echo(f"❌ Invalid selection. Please choose numbers between 1 and {len(recommendations)}.", err=True)
                continue
            
            selected = [recommendations[idx] for idx in indices]
            typer.echo(f"\n✅ Selected {len(selected)} model(s): {', '.join([m.name for m in selected])}")
            return selected
            
        except ValueError:
            typer.echo("❌ Invalid input. Please enter comma-separated numbers (e.g., 1,2,3).", err=True)
        except KeyboardInterrupt:
            typer.echo("\n\nSelection cancelled.")
            raise typer.Abort()

