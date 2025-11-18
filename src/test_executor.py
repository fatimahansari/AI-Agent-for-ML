"""Execute generated code files and collect metrics."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

console = Console()


class TestExecutor:
    """Execute generated Python code files and collect evaluation metrics."""
    
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []
    
    def execute_code(
        self,
        code_path: Path,
        model_name: str,
        timeout: int = 300,
    ) -> Dict[str, Any]:
        """Execute a generated code file and collect metrics.
        
        Args:
            code_path: Path to the Python code file
            model_name: Name of the model being evaluated
            timeout: Maximum execution time in seconds
            
        Returns:
            Dictionary with execution results and metrics
        """
        console.print(f"\n[bold cyan]Executing code for {model_name}...[/bold cyan]")
        
        try:
            # Execute the code - use absolute path and don't change cwd
            code_path_abs = code_path.resolve()
            result = subprocess.run(
                [sys.executable, str(code_path_abs)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            # Parse metrics from stdout
            metrics = self._parse_metrics(stdout, code_path)
            
            # Check for error patterns even if returncode is 0 (errors caught by try-except)
            error_patterns = [
                "error occurred",
                "exception",
                "traceback",
                "could not convert",
                "typeerror",
                "valueerror",
                "keyerror",
                "attributeerror",
                "indexerror",
                "failed",
            ]
            
            has_error = False
            error_message = ""
            
            # Check stdout and stderr for error patterns
            output_lower = (stdout + "\n" + stderr).lower()
            for pattern in error_patterns:
                if pattern in output_lower:
                    has_error = True
                    # Extract error message from output
                    if pattern in stderr.lower():
                        error_message = stderr
                        break
                    elif pattern in stdout.lower():
                        # Try to find the error line in stdout
                        for line in stdout.split("\n"):
                            if pattern in line.lower():
                                error_message = line.strip()
                                break
                        if not error_message:
                            error_message = stdout
                    break
            
            # If return code is not 0, definitely an error
            if result.returncode != 0:
                has_error = True
                if not error_message:
                    error_message = stderr or stdout or f"Exit code: {result.returncode}"
            
            # If we have an error and no metrics, it's a real error
            if has_error:
                if not metrics:
                    # Real error - no metrics produced
                    return {
                        "model_name": model_name,
                        "status": "error",
                        "return_code": result.returncode if result.returncode != 0 else -1,
                        "stdout": stdout,
                        "stderr": stderr,
                        "metrics": metrics,
                        "error_message": error_message,
                    }
                else:
                    # Error occurred but we still got metrics (partial success, but still consider it error)
                    # We want to fix it, so mark as error
                    return {
                        "model_name": model_name,
                        "status": "error",
                        "return_code": result.returncode if result.returncode != 0 else -1,
                        "stdout": stdout,
                        "stderr": stderr,
                        "metrics": metrics,
                        "error_message": error_message,
                    }
            
            # No error detected and we have metrics - success
            if metrics:
                return {
                    "model_name": model_name,
                    "status": "success",
                    "return_code": 0,
                    "stdout": stdout,
                    "stderr": stderr,
                    "metrics": metrics,
                }
            
            # No error detected but no metrics either - might be a problem
            if result.returncode == 0:
                # Check if output suggests code ran but didn't print metrics
                if len(stdout) < 50 and not stderr:
                    # Very little output, might have run silently
                    return {
                        "model_name": model_name,
                        "status": "error",
                        "return_code": -1,
                        "stdout": stdout,
                        "stderr": stderr or "No metrics found in output",
                        "metrics": metrics,
                        "error_message": "No metrics found in output",
                    }
            
            # Default: return as error if no clear success
            return {
                "model_name": model_name,
                "status": "error" if result.returncode != 0 else "success",
                "return_code": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "metrics": metrics,
                "error_message": error_message or "Unknown error",
            }
            
        except subprocess.TimeoutExpired:
            return {
                "model_name": model_name,
                "status": "timeout",
                "return_code": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {timeout} seconds",
                "metrics": {},
            }
        except Exception as e:
            return {
                "model_name": model_name,
                "status": "error",
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "metrics": {},
            }
    
    def _parse_metrics(self, output: str, code_path: Path) -> Dict[str, float]:
        """Parse evaluation metrics from code output.
        
        Looks for patterns like:
        - Accuracy: 0.85
        - RMSE: 0.42
        - etc.
        """
        metrics = {}
        
        # Common metric patterns
        metric_patterns = {
            r"accuracy[:\s=]+([0-9.]+)": "accuracy",
            r"precision[:\s=]+([0-9.]+)": "precision",
            r"recall[:\s=]+([0-9.]+)": "recall",
            r"f1[:\s\-]?score[:\s=]+([0-9.]+)": "f1_score",
            r"f1[:\s=]+([0-9.]+)": "f1_score",
            r"roc[-\s]?auc[:\s=]+([0-9.]+)": "roc_auc",
            r"mse[:\s=]+([0-9.]+)": "mse",
            r"rmse[:\s=]+([0-9.]+)": "rmse",
            r"mae[:\s=]+([0-9.]+)": "mae",
            r"r2[:\s=]+([0-9.]+)": "r2_score",
            r"r²[:\s=]+([0-9.]+)": "r2_score",
            r"r\^2[:\s=]+([0-9.]+)": "r2_score",
        }
        
        output_lower = output.lower()
        
        for pattern, metric_name in metric_patterns.items():
            matches = re.findall(pattern, output_lower, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[0])
                    metrics[metric_name] = value
                except ValueError:
                    continue
        
        # Also try to parse from common print formats
        lines = output.split("\n")
        for line in lines:
            for pattern, metric_name in metric_patterns.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        if metric_name not in metrics:  # Don't overwrite
                            metrics[metric_name] = value
                    except ValueError:
                        continue
        
        return metrics
    
    def display_results(self, results: List[Dict[str, Any]], problem_type: str) -> None:
        """Display execution results in a formatted table.
        
        Args:
            results: List of execution result dictionaries
            problem_type: Problem type (classification or regression)
        """
        console.print("\n[bold green]Model Evaluation Results[/bold green]")
        console.print("=" * 80)
        
        # Group by status
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] != "success"]
        
        # Display successful results
        if successful:
            table = Table(title="Model Performance Metrics")
            
            # Determine columns based on problem type
            if problem_type == "classification":
                columns = ["Model", "Status", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
            else:  # regression
                columns = ["Model", "Status", "RMSE", "MSE", "MAE", "R²"]
            
            for col in columns:
                table.add_column(col, justify="right" if col != "Model" else "left")
            
            for result in successful:
                metrics = result["metrics"]
                row = [result["model_name"], "✓ Success"]
                
                if problem_type == "classification":
                    row.extend([
                        f"{metrics.get('accuracy', 'N/A'):.4f}" if 'accuracy' in metrics else "N/A",
                        f"{metrics.get('precision', 'N/A'):.4f}" if 'precision' in metrics else "N/A",
                        f"{metrics.get('recall', 'N/A'):.4f}" if 'recall' in metrics else "N/A",
                        f"{metrics.get('f1_score', 'N/A'):.4f}" if 'f1_score' in metrics else "N/A",
                        f"{metrics.get('roc_auc', 'N/A'):.4f}" if 'roc_auc' in metrics else "N/A",
                    ])
                else:
                    row.extend([
                        f"{metrics.get('rmse', 'N/A'):.4f}" if 'rmse' in metrics else "N/A",
                        f"{metrics.get('mse', 'N/A'):.4f}" if 'mse' in metrics else "N/A",
                        f"{metrics.get('mae', 'N/A'):.4f}" if 'mae' in metrics else "N/A",
                        f"{metrics.get('r2_score', 'N/A'):.4f}" if 'r2_score' in metrics else "N/A",
                    ])
                
                table.add_row(*row)
            
            console.print(table)
        
        # Display failed results
        if failed:
            console.print("\n[bold red]Failed Executions:[/bold red]")
            for result in failed:
                console.print(f"\n[red]✗ {result['model_name']}[/red]")
                console.print(f"Status: {result['status']}")
                if result.get("stderr"):
                    console.print(f"[dim]Error: {result['stderr'][:200]}[/dim]")
                if result.get("stdout"):
                    console.print(f"[dim]Output: {result['stdout'][:200]}[/dim]")
        
        # Display full output for debugging
        console.print("\n[bold]Detailed Output:[/bold]")
        for result in results:
            console.print(f"\n[bold cyan]{result['model_name']}:[/bold cyan]")
            if result.get("stdout"):
                console.print("[dim]" + result["stdout"][:500] + "[/dim]")
            if result.get("stderr"):
                console.print("[red][dim]" + result["stderr"][:500] + "[/dim][/red]")
