"""
Memory Manager for Reasoning Agent Brain
Stores and retrieves:
- Dataset analysis (from report.json)
- Model parameters used
- Evaluation results generated
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib


class MemoryManager:
    """Manages persistent memory for Reasoning Agent operations."""
    
    def __init__(self, brain_dir: Optional[Path] = None):
        """
        Initialize the Memory Manager.
        
        Args:
            brain_dir: Directory to store brain data. Defaults to 'brain/memories' relative to this file.
        """
        if brain_dir is None:
            brain_dir = Path(__file__).parent / "memories"
        self.brain_dir = Path(brain_dir)
        self.brain_dir.mkdir(parents=True, exist_ok=True)
        
        # Index file to track all memories
        self.index_file = self.brain_dir / "index.json"
        self._load_index()
    
    def _load_index(self) -> None:
        """Load the index of all stored memories."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    self.index = json.load(f)
            except Exception:
                self.index = {"experiments": []}
        else:
            self.index = {"experiments": []}
    
    def _save_index(self) -> None:
        """Save the index of all stored memories."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _generate_experiment_id(self, dataset_path: str, target_column: str) -> str:
        """Generate a unique experiment ID based on dataset and target."""
        # Create a hash from dataset path and target column
        content = f"{dataset_path}:{target_column}"
        hash_obj = hashlib.md5(content.encode())
        return hash_obj.hexdigest()[:12]
    
    def store_dataset_analysis(
        self,
        dataset_path: str,
        target_column: str,
        report_data: Dict[str, Any],
        report_path: Optional[str] = None
    ) -> str:
        """
        Store dataset analysis from report.json.
        
        Args:
            dataset_path: Path to the dataset file
            target_column: Name of the target column
            report_data: Dictionary containing report.json data
            report_path: Optional path to the original report.json file
            
        Returns:
            Experiment ID for this dataset/target combination
        """
        experiment_id = self._generate_experiment_id(dataset_path, target_column)
        experiment_dir = self.brain_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Store dataset analysis
        analysis_file = experiment_dir / "dataset_analysis.json"
        analysis_data = {
            "experiment_id": experiment_id,
            "dataset_path": str(dataset_path),
            "target_column": target_column,
            "report_path": str(report_path) if report_path else None,
            "timestamp": datetime.now().isoformat(),
            "analysis": report_data
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        # Update index
        if not any(exp["experiment_id"] == experiment_id for exp in self.index["experiments"]):
            self.index["experiments"].append({
                "experiment_id": experiment_id,
                "dataset_path": str(dataset_path),
                "target_column": target_column,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            })
        else:
            # Update last_updated
            for exp in self.index["experiments"]:
                if exp["experiment_id"] == experiment_id:
                    exp["last_updated"] = datetime.now().isoformat()
                    break
        
        self._save_index()
        return experiment_id
    
    def store_model_parameters(
        self,
        experiment_id: str,
        model_name: str,
        model_class: str,
        hyperparameters: Dict[str, Any],
        problem_type: str,
        test_size: float = 0.2,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store model parameters used in an experiment.
        
        Args:
            experiment_id: Experiment ID from store_dataset_analysis
            model_name: Name of the model (e.g., "Random Forest")
            model_class: Model class name (e.g., "RandomForestClassifier")
            hyperparameters: Dictionary of hyperparameters used
            problem_type: Problem type ("classification" or "regression")
            test_size: Test set size fraction
            additional_info: Optional additional information
        """
        experiment_dir = self.brain_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Load existing models or create new
        models_file = experiment_dir / "model_parameters.json"
        if models_file.exists():
            with open(models_file, 'r') as f:
                models_data = json.load(f)
        else:
            models_data = {"models": []}
        
        # Check if this model already exists
        model_exists = False
        for model in models_data["models"]:
            if model["model_name"] == model_name:
                # Update existing model
                model.update({
                    "model_class": model_class,
                    "hyperparameters": hyperparameters,
                    "problem_type": problem_type,
                    "test_size": test_size,
                    "updated_at": datetime.now().isoformat(),
                    "additional_info": additional_info or {}
                })
                model_exists = True
                break
        
        if not model_exists:
            # Add new model
            models_data["models"].append({
                "model_name": model_name,
                "model_class": model_class,
                "hyperparameters": hyperparameters,
                "problem_type": problem_type,
                "test_size": test_size,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "additional_info": additional_info or {}
            })
        
        with open(models_file, 'w') as f:
            json.dump(models_data, f, indent=2)
    
    def store_evaluation_results(
        self,
        experiment_id: str,
        model_name: str,
        metrics: Dict[str, Any],
        execution_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store evaluation results for a model.
        
        Args:
            experiment_id: Experiment ID from store_dataset_analysis
            model_name: Name of the model
            metrics: Dictionary of evaluation metrics
            execution_info: Optional execution information (status, stdout, stderr, etc.)
        """
        experiment_dir = self.brain_dir / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Load existing results or create new
        results_file = experiment_dir / "evaluation_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results_data = json.load(f)
        else:
            results_data = {"evaluations": []}
        
        # Add new evaluation result
        evaluation = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "execution_info": execution_info or {}
        }
        
        results_data["evaluations"].append(evaluation)
        
        # Keep only last 10 evaluations per model (to avoid file bloat)
        model_evals = [e for e in results_data["evaluations"] if e["model_name"] == model_name]
        if len(model_evals) > 10:
            # Sort by timestamp and keep only latest 10
            model_evals.sort(key=lambda x: x["timestamp"], reverse=True)
            results_data["evaluations"] = [
                e for e in results_data["evaluations"]
                if e["model_name"] != model_name
            ] + model_evals[:10]
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def get_experiment_history(
        self,
        dataset_path: Optional[str] = None,
        target_column: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete experiment history.
        
        Args:
            dataset_path: Optional dataset path to search for
            target_column: Optional target column to search for
            experiment_id: Optional specific experiment ID
            
        Returns:
            Dictionary containing dataset analysis, model parameters, and evaluation results
        """
        # Find experiment ID
        if experiment_id is None:
            if dataset_path and target_column:
                experiment_id = self._generate_experiment_id(dataset_path, target_column)
            else:
                return None
        
        experiment_dir = self.brain_dir / experiment_id
        if not experiment_dir.exists():
            return None
        
        history = {
            "experiment_id": experiment_id,
            "dataset_analysis": None,
            "model_parameters": None,
            "evaluation_results": None
        }
        
        # Load dataset analysis
        analysis_file = experiment_dir / "dataset_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                history["dataset_analysis"] = json.load(f)
        
        # Load model parameters
        models_file = experiment_dir / "model_parameters.json"
        if models_file.exists():
            with open(models_file, 'r') as f:
                history["model_parameters"] = json.load(f)
        
        # Load evaluation results
        results_file = experiment_dir / "evaluation_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                history["evaluation_results"] = json.load(f)
        
        return history
    
    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """
        Get list of all stored experiments.
        
        Returns:
            List of experiment summaries
        """
        return self.index.get("experiments", [])
    
    def compare_before_after(
        self,
        experiment_id: str,
        model_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Compare evaluation results before and after improvements.
        Useful for comparing original vs improved model performance.
        
        Args:
            experiment_id: Experiment ID
            model_name: Model name to compare
            
        Returns:
            Dictionary with before/after comparison if available
        """
        history = self.get_experiment_history(experiment_id=experiment_id)
        if not history or not history.get("evaluation_results"):
            return None
        
        evaluations = history["evaluation_results"]["evaluations"]
        model_evals = [e for e in evaluations if e["model_name"] == model_name]
        
        if len(model_evals) < 2:
            return None
        
        # Sort by timestamp
        model_evals.sort(key=lambda x: x["timestamp"])
        
        # First evaluation is "before", last is "after"
        before = model_evals[0]
        after = model_evals[-1]
        
        comparison = {
            "model_name": model_name,
            "before": {
                "timestamp": before["timestamp"],
                "metrics": before["metrics"]
            },
            "after": {
                "timestamp": after["timestamp"],
                "metrics": after["metrics"]
            },
            "improvements": {}
        }
        
        # Calculate improvements
        before_metrics = before["metrics"]
        after_metrics = after["metrics"]
        
        for metric_name in set(before_metrics.keys()) | set(after_metrics.keys()):
            if metric_name in before_metrics and metric_name in after_metrics:
                before_val = before_metrics[metric_name]
                after_val = after_metrics[metric_name]
                
                # For metrics where higher is better (accuracy, r2_score, etc.)
                if metric_name in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "r2_score"]:
                    if before_val > 0:
                        improvement_pct = ((after_val - before_val) / before_val) * 100
                        comparison["improvements"][metric_name] = {
                            "before": before_val,
                            "after": after_val,
                            "improvement_pct": improvement_pct,
                            "improved": after_val > before_val
                        }
                # For metrics where lower is better (mse, rmse, mae)
                elif metric_name in ["mse", "rmse", "mae"]:
                    if before_val > 0:
                        improvement_pct = ((before_val - after_val) / before_val) * 100
                        comparison["improvements"][metric_name] = {
                            "before": before_val,
                            "after": after_val,
                            "improvement_pct": improvement_pct,
                            "improved": after_val < before_val
                        }
        
        return comparison
    
    def load_report_json(self, report_path: str) -> Optional[Dict[str, Any]]:
        """
        Load and return report.json data.
        
        Args:
            report_path: Path to report.json file
            
        Returns:
            Dictionary containing report data, or None if file doesn't exist
        """
        report_file = Path(report_path)
        if not report_file.exists():
            return None
        
        try:
            with open(report_file, 'r') as f:
                return json.load(f)
        except Exception:
            return None

