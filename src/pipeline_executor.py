"""Orchestrate the complete model recommendation and evaluation pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Dict, Any

from rich.console import Console
from rich.markdown import Markdown

from src.code_generator import CodeGenerator
from src.model_compatibility import ModelCompatibilityChecker
from src.model_recommender import DatasetMetadata, ModelRecommenderAgent
from src.test_executor import TestExecutor

console = Console()


class PipelineExecutor:
    """Orchestrate model recommendation, selection, code generation, and execution."""
    
    def __init__(self) -> None:
        self.recommender = ModelRecommenderAgent()
        self.code_generator = CodeGenerator()
        self.test_executor = TestExecutor()
    
    def execute_pipeline(
        self,
        *,
        metadata: DatasetMetadata,
        dataset_path: Path,
        context: Optional[str] = None,
        test_size: float = 0.2,
        min_models: int = 1,
        max_models: int = 3,
        code_dir: Path = Path("generated_code"),
        excluded_models: Optional[List[str]] = None,
    ) -> List[dict]:
        """Execute the complete pipeline.
        
        Args:
            metadata: Dataset metadata
            dataset_path: Path to the dataset file
            context: Extra context for recommendations
            test_size: Fraction of data for testing
            min_models: Minimum number of models to select
            max_models: Maximum number of models to select
            code_dir: Directory to save generated code
            excluded_models: List of model names to exclude from recommendations
            
        Returns:
            List of execution result dictionaries
        """
        excluded_models = excluded_models or []
        
        # Step 1: Get model recommendations
        console.print("\n[bold blue]Step 1: Getting model recommendations from LLM...[/bold blue]")
        if excluded_models:
            exclusion_context = f"{context or ''} (Exclude these previously suggested models: {', '.join(excluded_models)})"
        else:
            exclusion_context = context
        recommendations = self.recommender.recommend(metadata=metadata, context=exclusion_context)
        
        # Extract model names and filter incompatible ones and excluded ones
        compatibility_checker = ModelCompatibilityChecker(metadata)
        model_options = self._extract_model_names(recommendations)
        
        # Filter compatible models and exclude previously suggested models
        compatible_models = []
        incompatible_models = []
        excluded_shown = []
        
        for model_name in model_options:
            # Check if model should be excluded
            if any(excluded.lower() in model_name.lower() or model_name.lower() in excluded.lower() 
                   for excluded in excluded_models):
                excluded_shown.append(model_name)
                continue
            
            # Check compatibility
            is_compat, reason = compatibility_checker.is_model_compatible(model_name)
            if is_compat:
                compatible_models.append(model_name)
            else:
                incompatible_models.append((model_name, reason))
        
        # Show excluded models if any
        if excluded_shown:
            console.print("\n[bold yellow]⚠ Previously Suggested Models (excluded):[/bold yellow]")
            for model_name in excluded_shown:
                console.print(f"[dim]  ⊘ {model_name}[/dim]")
        
        # Display recommendations with compatibility info
        console.print("\n[bold green]Recommended Models:[/bold green]")
        console.print(Markdown(recommendations))
        
        # Show incompatible models if any
        if incompatible_models:
            console.print("\n[bold yellow]⚠ Incompatible Models (filtered out):[/bold yellow]")
            for model_name, reason in incompatible_models:
                console.print(f"[dim]  ✗ {model_name}: {reason}[/dim]")
        
        # Show compatible models
        if not compatible_models:
            console.print("\n[red]No compatible models found for this dataset![/red]")
            return []
        
        console.print(f"\n[bold green]✓ {len(compatible_models)} compatible model(s) found[/bold green]")
        
        # Step 2: Prompt user for model selection (only from compatible models)
        console.print(f"\n[bold yellow]Step 2: Please select {min_models}-{max_models} models to train and evaluate.[/bold yellow]")
        selected_models = self._prompt_model_selection_from_list(compatible_models, min_models, max_models)
        
        if not selected_models:
            console.print("[red]No models selected. Exiting.[/red]")
            return []
        
        # Step 3: Generate validation code for all selected models
        console.print(f"\n[bold blue]Step 3: Generating validation code for {len(selected_models)} model(s)...[/bold blue]")
        
        # Clean model names
        clean_model_names = [re.sub(r'\*\*|\*|#', '', name).strip() for name in selected_models]
        
        # Generate validation code file for all selected models
        console.print("[cyan]Generating validation testing script...[/cyan]")
        validation_code = self.code_generator.generate_validation_code(
            metadata=metadata,
            dataset_path=dataset_path,
            selected_models=clean_model_names,
            test_size=test_size,
        )
        
        # Save validation code
        code_dir.mkdir(parents=True, exist_ok=True)
        validation_file = code_dir / "validate_models.py"
        with open(validation_file, "w", encoding="utf-8") as f:
            f.write(validation_code)
        console.print(f"[green]✓ Validation code saved to {validation_file}[/green]")
        
        # Step 4: Execute validation code
        console.print(f"\n[bold blue]Step 4: Executing validation code - Testing models with 80-20 split...[/bold blue]")
        result = self.test_executor.execute_code(validation_file, "validation")
        
        # Step 5: Parse validation metrics from output and create per-model results
        results = []
        if result["status"] == "success":
            # Parse validation metrics for each model from stdout
            stdout = result.get("stdout", "")
            results = self._parse_validation_results(stdout, clean_model_names, metadata.problem_type)
            
            # Display validation metrics
            console.print("\n[bold green]VALIDATION TESTING METRICS (80-20 Split)[/bold green]")
            self.test_executor.display_results(results, metadata.problem_type)
        else:
            console.print("[red]✗ Validation code execution failed[/red]")
            error_msg = result.get("error_message", "") or result.get("stderr", "") or result.get("stdout", "")
            if error_msg:
                console.print(f"[dim]Error: {error_msg[:500]}[/dim]")
            results = [result]
        
        # Step 6: Prompt user for satisfaction
        return self._handle_satisfaction_feedback(
            results=results,
            selected_models=selected_models,
            metadata=metadata,
            dataset_path=dataset_path,
            code_dir=code_dir,
            context=context,
            test_size=test_size,
            min_models=min_models,
            max_models=max_models,
            excluded_models=[],
        )
    
    def _parse_validation_results(self, stdout: str, selected_models: List[str], problem_type: str) -> List[dict]:
        """Parse validation metrics from deployment code output.
        
        Args:
            stdout: Standard output from deployment code execution
            selected_models: List of selected model names
            problem_type: Problem type (classification or regression)
            
        Returns:
            List of result dictionaries with metrics for each model
        """
        results = []
        lines = stdout.split("\n")
        
        for model_name in selected_models:
            metrics = {}
            in_model_section = False
            model_section_lines = []
            
            # Find the validation section for this model
            model_pattern = re.compile(rf"VALIDATION TESTING\s*-\s*{re.escape(model_name)}", re.IGNORECASE)
            metrics_pattern = re.compile(rf"VALIDATION METRICS\s*-\s*{re.escape(model_name)}", re.IGNORECASE)
            
            for i, line in enumerate(lines):
                if metrics_pattern.search(line):
                    # Found metrics section for this model, collect next few lines
                    for j in range(i, min(i + 10, len(lines))):
                        model_section_lines.append(lines[j])
                    break
            
            # Parse metrics from the collected lines
            if model_section_lines:
                section_text = "\n".join(model_section_lines)
                metrics = self.test_executor._parse_metrics(section_text, None)
            
            # If no metrics found in section, try parsing whole output (fallback)
            if not metrics:
                metrics = self.test_executor._parse_metrics(stdout, None)
            
            # Create result dictionary
            result = {
                "model_name": model_name,
                "status": "success" if metrics else "unknown",
                "metrics": metrics,
                "stdout": stdout,
                "stderr": "",
            }
            results.append(result)
        
        return results
    
    def _extract_model_names(self, recommendations: str) -> List[str]:
        """Extract model names from recommendations text.
        
        Args:
            recommendations: Markdown text with model recommendations
            
        Returns:
            List of model names
        """
        lines = recommendations.split("\n")
        model_options = []
        
        for line in lines:
            line = line.strip()
            # Look for numbered items (1., 2., etc.) or headings with numbers followed by a period
            if re.match(r'^(\d+\.|#+\s|\*\*?\d+\.)', line):
                # Extract the model name after the number/heading marker
                model_name = re.sub(r'^\d+\.\s*', '', line)
                # Remove markdown bold markers (**text**)
                model_name = re.sub(r'\*\*([^*]+)\*\*', r'\1', model_name)
                # Remove single asterisks
                model_name = re.sub(r'\*', '', model_name)
                # Remove heading markers
                model_name = re.sub(r'^#+\s*', '', model_name)
                # Take only the first part (before colon, dash, or newline)
                model_name = model_name.split(":")[0].split("-")[0].split("\n")[0].strip()
                # Clean up leading/trailing whitespace and special chars
                model_name = re.sub(r'^[\s#*]+|[\s#*]+$', '', model_name)
                # Skip if it's too short or looks like a header
                if model_name and len(model_name) > 2 and not model_name.lower().startswith(('model recommendations', 'given the', 'conclusion')):
                    model_options.append(model_name)
        
        return model_options
    
    def _prompt_model_selection_from_list(
        self,
        model_options: List[str],
        min_models: int,
        max_models: int,
    ) -> List[str]:
        """Prompt user to select models from a list.
        
        Args:
            model_options: List of model names to choose from
            min_models: Minimum number of models to select
            max_models: Maximum number of models to select
            
        Returns:
            List of selected model names
        """
        if not model_options:
            console.print("\n[dim]Could not auto-detect model names from recommendations.[/dim]")
            console.print("[dim]Please enter model names manually, one per line.[/dim]")
            
            selected = []
            for i in range(min_models, max_models + 1):
                if i > min_models:
                    console.print(f"\nEnter model #{i} (or press Enter to finish): ", end="")
                else:
                    console.print(f"\nEnter model #{i}: ", end="")
                
                model_name = input().strip()
                if not model_name:
                    if len(selected) >= min_models:
                        break
                    continue
                selected.append(model_name)
            
            return selected if len(selected) >= min_models else []
        
        # Display numbered options (only compatible models shown)
        console.print("\n[bold]Available models (compatible):[/bold]")
        for i, model in enumerate(model_options, 1):
            console.print(f"  {i}. {model}")
        
        console.print(f"\n[bold]Select {min_models}-{max_models} model(s) by entering numbers separated by commas:[/bold]")
        console.print("[dim](e.g., '1,2,3' or '1 2 3')[/dim]")
        
        while True:
            try:
                user_input = input().strip()
                if not user_input:
                    continue
                
                # Parse input (handle comma or space separated)
                indices = []
                for part in user_input.replace(",", " ").split():
                    indices.append(int(part) - 1)  # Convert to 0-based
                
                if len(indices) < min_models:
                    console.print(f"[red]Please select at least {min_models} model(s).[/red]")
                    continue
                
                if len(indices) > max_models:
                    console.print(f"[red]Please select at most {max_models} model(s).[/red]")
                    continue
                
                # Get selected models
                selected = []
                for idx in indices:
                    if 0 <= idx < len(model_options):
                        selected.append(model_options[idx])
                    else:
                        console.print(f"[red]Invalid index: {idx + 1}[/red]")
                        break
                
                if len(selected) == len(indices):
                    return selected
                    
            except ValueError:
                console.print("[red]Please enter valid numbers separated by commas or spaces.[/red]")
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled by user.[/yellow]")
                return []
    
    def _prompt_model_selection(
        self,
        recommendations: str,
        min_models: int,
        max_models: int,
    ) -> List[str]:
        """Prompt user to select models from recommendations (deprecated - use _prompt_model_selection_from_list).
        
        Args:
            recommendations: Markdown text with model recommendations
            min_models: Minimum number of models to select
            max_models: Maximum number of models to select
            
        Returns:
            List of selected model names
        """
        model_options = self._extract_model_names(recommendations)
        return self._prompt_model_selection_from_list(model_options, min_models, max_models)
    
    def _handle_satisfaction_feedback(
        self,
        *,
        results: List[dict],
        selected_models: List[str],
        metadata: DatasetMetadata,
        dataset_path: Path,
        code_dir: Path,
        context: Optional[str],
        test_size: float,
        min_models: int,
        max_models: int,
        excluded_models: List[str],
    ) -> List[dict]:
        """Handle user satisfaction feedback after displaying results.
        
        Args:
            results: List of execution result dictionaries
            selected_models: List of selected model names
            metadata: Dataset metadata
            dataset_path: Path to the dataset file
            code_dir: Directory to save generated code
            context: Extra context for recommendations
            test_size: Fraction of data for testing
            min_models: Minimum number of models to select
            max_models: Maximum number of models to select
            excluded_models: List of previously excluded models
            
        Returns:
            List of execution result dictionaries (may be from new cycle)
        """
        # Filter only successful results
        successful_results = [r for r in results if r["status"] == "success"]
        
        if not successful_results:
            console.print("\n[red]No successful model evaluations found. Cannot proceed with satisfaction feedback.[/red]")
            return results
        
        # Prompt user for satisfaction
        console.print("\n[bold yellow]" + "=" * 80 + "[/bold yellow]")
        console.print("\n[bold cyan]Are you satisfied with the results?[/bold cyan]")
        console.print("\n[bold]Options:[/bold]")
        console.print("  1. Yes, I'm satisfied - Generate full training code and save model (.pkl)")
        console.print("  2. No, I'm not satisfied - Show more options")
        console.print("  3. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1/2/3): ").strip()
                
                if choice == "1":
                    # Generate full training code for satisfied user
                    return self._generate_full_training_code(
                        successful_results=successful_results,
                        selected_models=selected_models,
                        metadata=metadata,
                        dataset_path=dataset_path,
                        code_dir=code_dir,
                    )
                
                elif choice == "2":
                    # Handle not satisfied option
                    return self._handle_not_satisfied(
                        results=results,
                        selected_models=selected_models,
                        metadata=metadata,
                        dataset_path=dataset_path,
                        code_dir=code_dir,
                        context=context,
                        test_size=test_size,
                        min_models=min_models,
                        max_models=max_models,
                        excluded_models=excluded_models + selected_models,
                    )
                
                elif choice == "3":
                    console.print("\n[yellow]Exiting...[/yellow]")
                    return results
                
                else:
                    console.print("[red]Invalid choice. Please enter 1, 2, or 3.[/red]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled by user.[/yellow]")
                return results
    
    def _generate_full_training_code(
        self,
        *,
        successful_results: List[dict],
        selected_models: List[str],
        metadata: DatasetMetadata,
        dataset_path: Path,
        code_dir: Path,
    ) -> List[dict]:
        """Generate deployment code for the best performing model based on validation metrics.
        
        Args:
            successful_results: List of successful execution results with metrics
            selected_models: List of selected model names
            metadata: Dataset metadata
            dataset_path: Path to the dataset file
            code_dir: Directory to save generated code
            
        Returns:
            List of execution result dictionaries
        """
        console.print("\n[bold blue]Identifying best performing model from validation metrics...[/bold blue]")
        
        # Find best performing model based on metrics
        best_model = self._find_best_model(successful_results, metadata.problem_type)
        
        if not best_model:
            console.print("[red]Could not determine best model. Please select manually.[/red]")
            # Fallback to first successful model
            best_model = successful_results[0]["model_name"]
        
        console.print(f"[green]✓ Best performing model: {best_model}[/green]")
        
        # Generate deployment code for best model
        console.print(f"\n[bold blue]Generating deployment code for best model: {best_model}...[/bold blue]")
        
        # Clean model name
        clean_model_name = re.sub(r'\*\*|\*|#', '', best_model).strip()
        
        # Generate deployment code
        deployment_code = self.code_generator.generate_deployment_code(
            metadata=metadata,
            dataset_path=dataset_path,
            model_name=clean_model_name,
        )
        
        # Save deployment code
        code_dir.mkdir(parents=True, exist_ok=True)
        deployment_file = code_dir / "deploy_best_model.py"
        with open(deployment_file, "w", encoding="utf-8") as f:
            f.write(deployment_code)
        console.print(f"[green]✓ Deployment code saved to {deployment_file}[/green]")
        
        # Execute deployment code
        console.print(f"\n[bold blue]Executing deployment code - Training {best_model} on full dataset and saving PKL...[/bold blue]")
        result = self.test_executor.execute_code(deployment_file, f"deployment_{best_model}")
        
        if result["status"] == "success":
            console.print(f"[green]✓ {best_model} trained on full dataset and saved as PKL file![/green]")
            console.print(f"[dim]Model saved in: models/ directory[/dim]")
        else:
            console.print(f"[red]✗ Deployment code execution failed[/red]")
            error_msg = result.get("error_message", "") or result.get("stderr", "") or result.get("stdout", "")
            if error_msg:
                console.print(f"[dim]Error: {error_msg[:500]}[/dim]")
        
        console.print("\n[bold green]✓ Deployment training completed![/bold green]")
        
        return successful_results
    
    def _find_best_model(self, results: List[dict], problem_type: str) -> Optional[str]:
        """Find the best performing model from validation results.
        
        Args:
            results: List of execution result dictionaries with metrics
            problem_type: Problem type (classification or regression)
            
        Returns:
            Name of the best performing model, or None if cannot determine
        """
        if not results:
            return None
        
        if problem_type == "classification":
            # For classification: use accuracy as primary metric, F1-score as secondary
            best_score = -1
            best_model = None
            
            for result in results:
                metrics = result.get("metrics", {})
                if "accuracy" in metrics:
                    score = metrics["accuracy"]
                    # If accuracy is same, prefer higher F1-score
                    if metrics.get("f1_score"):
                        score = (score + metrics["f1_score"]) / 2
                    
                    if score > best_score:
                        best_score = score
                        best_model = result["model_name"]
            
            return best_model
        else:  # regression
            # For regression: use R² as primary metric, RMSE as secondary
            best_score = -float('inf')
            best_model = None
            
            for result in results:
                metrics = result.get("metrics", {})
                if "r2_score" in metrics or "r2" in metrics:
                    r2 = metrics.get("r2_score") or metrics.get("r2", 0)
                    # Higher R² is better
                    if r2 > best_score:
                        best_score = r2
                        best_model = result["model_name"]
                elif "rmse" in metrics:
                    # If no R², use negative RMSE (lower is better)
                    rmse = metrics["rmse"]
                    score = -rmse
                    if score > best_score:
                        best_score = score
                        best_model = result["model_name"]
            
            return best_model
    
    def _handle_not_satisfied(
        self,
        *,
        results: List[dict],
        selected_models: List[str],
        metadata: DatasetMetadata,
        dataset_path: Path,
        code_dir: Path,
        context: Optional[str],
        test_size: float,
        min_models: int,
        max_models: int,
        excluded_models: List[str],
    ) -> List[dict]:
        """Handle user feedback when not satisfied.
        
        Args:
            results: List of execution result dictionaries
            selected_models: List of selected model names
            metadata: Dataset metadata
            dataset_path: Path to the dataset file
            code_dir: Directory to save generated code
            context: Extra context for recommendations
            test_size: Fraction of data for testing
            min_models: Minimum number of models to select
            max_models: Maximum number of models to select
            excluded_models: List of previously excluded models
            
        Returns:
            List of execution result dictionaries (may be from new cycle)
        """
        console.print("\n[bold cyan]What would you like to do?[/bold cyan]")
        console.print("\n[bold]Options:[/bold]")
        console.print("  1. Use the same models but improve them using the improvement agent")
        console.print("  2. Change the model selection (get new recommendations excluding current models)")
        console.print("  3. Cancel")
        
        while True:
            try:
                choice = input("\nEnter your choice (1/2/3): ").strip()
                
                if choice == "1":
                    console.print("\n[yellow]Improvement agent integration will be available in a future update.[/yellow]")
                    console.print("[yellow]Exiting for now...[/yellow]")
                    return results
                
                elif choice == "2":
                    # Repeat recommendation cycle with excluded models
                    console.print("\n[bold blue]Starting new recommendation cycle with different models...[/bold blue]")
                    return self.execute_pipeline(
                        metadata=metadata,
                        dataset_path=dataset_path,
                        context=context,
                        test_size=test_size,
                        min_models=min_models,
                        max_models=max_models,
                        code_dir=code_dir,
                        excluded_models=excluded_models,
                    )
                
                elif choice == "3":
                    console.print("\n[yellow]Cancelled.[/yellow]")
                    return results
                
                else:
                    console.print("[red]Invalid choice. Please enter 1, 2, or 3.[/red]")
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled by user.[/yellow]")
                return results
