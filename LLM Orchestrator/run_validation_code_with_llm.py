#Usage: python LLM Orchestrator/run_validation_code_with_llm.py
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
LLM_MODEL = "phi4"
LLM_ENDPOINT = "http://127.0.0.1:11434"
MAX_RETRIES = 3

llm = ChatOllama(model=LLM_MODEL, base_url=LLM_ENDPOINT, temperature=0.0)
parser = StrOutputParser()

FIX_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Python ML engineer. Fix the provided validation/testing code "
            "without changing the model or evaluation logic. Maintain the exact model implementation "
            "and metrics calculation. Use the dataset context and report to reason about required "
            "columns and data types. Return only valid Python code that prevents "
            "the reported error from reoccurring.",
        ),
        (
            "user",
            "Model recommendations context:\n{model_recommendations}\n\n"
            "Dataset context:\n{dataset_context}\n\n"
            "Dataset report:\n{dataset_report}\n\n"
            "Current code:\n{code}\n\n"
            "Execution error traceback:\n{error}\n\n"
            "Return only corrected Python code.",
        ),
    ]
)


# --------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------
def extract_code_snippet(content: str) -> str:
    """
    Extract the first fenced code block from the LLM response.
    Falls back to stripping fences if no explicit block is found.
    """
    if not content:
        return content

    match = re.search(r"```(?:[\w+-]+)?\s*([\s\S]*?)```", content)
    if match:
        return match.group(1).strip()

    return strip_code_fences(content)


def strip_code_fences(content: str) -> str:
    """Remove markdown code fences from LLM output."""
    if not content:
        return content
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        newline_idx = cleaned.find("\n")
        if newline_idx != -1:
            maybe_lang = cleaned[:newline_idx].strip().lower()
            if maybe_lang.isalpha():
                cleaned = cleaned[newline_idx + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def load_memory(memory_file: Path) -> dict:
    if not memory_file.exists():
        raise FileNotFoundError(
            f"Orchestrator memory not found at {memory_file}. "
            "Run the LLM orchestrator workflow agent first."
        )
    with memory_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base_dir: Path, stored_path: Optional[str], fallback_name: Optional[str]) -> Path:
    if stored_path:
        stored = Path(stored_path)
        return stored if stored.is_absolute() else (base_dir / stored).resolve()
    if fallback_name:
        fallback = Path(fallback_name)
        if fallback.is_absolute():
            return fallback
        return (base_dir / fallback).resolve()
    raise FileNotFoundError("Unable to determine required path from memory.")


def fix_code_with_llm(
    code: str, 
    error: str, 
    model_recommendations: str, 
    dataset_context: str,
    dataset_report: str
) -> str:
    chain = FIX_PROMPT | llm | parser
    fixed = chain.invoke(
        {
            "code": code,
            "error": error,
            "model_recommendations": model_recommendations,
            "dataset_context": dataset_context,
            "dataset_report": dataset_report,
        }
    )
    return extract_code_snippet(fixed)


# --------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------
def main():
    base_dir = Path(__file__).parent
    memory_file = base_dir / "orchestrator_memory.json"
    memory_data = load_memory(memory_file)

    # Get master memory path to load additional context
    master_memory_path = memory_data.get("master_memory_path")
    if not master_memory_path:
        raise FileNotFoundError("'master_memory_path' not found in orchestrator_memory.json")
    
    master_memory_path = Path(master_memory_path)
    if not master_memory_path.exists():
        raise FileNotFoundError(f"Master memory file not found: {master_memory_path}")
    
    # Load master memory for dataset context and report
    with master_memory_path.open("r", encoding="utf-8") as f:
        master_memory = json.load(f)
    
    # Prepare context for LLM fixes
    dataset_context = json.dumps({
        "dataset_name": memory_data.get("dataset_name"),
        "target_column": memory_data.get("target_column"),
        "context_of_dataset": memory_data.get("context_of_dataset"),
        "dataset_path": master_memory.get("dataset_path"),
        "processed_output_path": master_memory.get("preprocessing", {}).get("processed_output_path"),
    }, indent=2)
    
    dataset_report = json.dumps(master_memory.get("preprocessing", {}).get("report", {}), indent=2)
    model_recommendations = json.dumps(memory_data.get("model_recommendations_structured", []), indent=2)

    # Get generated code paths
    generated_code_paths = memory_data.get("generated_code_paths", [])
    
    # Fallback: try to get from file_history
    if not generated_code_paths:
        file_history = memory_data.get("file_history", {})
        generated_code_paths = [
            file_history[key] 
            for key in sorted(file_history.keys()) 
            if key.startswith("validation_code_")
        ]
    
    # Fallback: try to find files in model_testing folder
    if not generated_code_paths:
        model_testing_dir = base_dir / "model_testing"
        if model_testing_dir.exists():
            generated_code_paths = [
                str(f.resolve()) 
                for f in model_testing_dir.glob("*_test.py")
            ]
    
    if not generated_code_paths:
        raise FileNotFoundError(
            "No validation/testing code files found. "
            "Run model_vc_gen.py or the LLM orchestrator workflow agent first."
        )

    print(f"[INFO] Found {len(generated_code_paths)} validation/testing code file(s) to run\n")

    # Prepare structure to store metrics
    structured_recs = memory_data.get("model_recommendations_structured", [])
    selected_models = memory_data.get("selected_models_for_validation", [])
    validation_metrics = memory_data.get("validation_metrics", {})

    # Run each validation code file
    for idx, code_path_str in enumerate(generated_code_paths):
        code_path = Path(code_path_str)
        
        if not code_path.exists():
            print(f"[WARN] Code file not found: {code_path}. Skipping.")
            continue
        
        # Determine model name from selected_models_for_validation first (matches the order of generated_code_paths)
        model_name = None
        if idx < len(selected_models):
            model_name = selected_models[idx]
        # Fallback: try to derive from code file name
        if not model_name:
            # Remove common suffixes like "_test" from filename
            code_name = code_path.stem.replace("_test", "").replace("_validation", "")
            # Try to match with any recommended model name
            for rec in structured_recs:
                rec_name = rec.get("Model Recommended", "")
                # Normalize names for comparison (lowercase, remove spaces/special chars)
                if rec_name.lower().replace(" ", "_").replace("-", "_") == code_name.lower():
                    model_name = rec_name
                    break
        # Final fallback: use code path stem
        if not model_name:
            model_name = code_path.stem
        
        print(f"[INFO] Running validation/testing code: {code_path.name} (model: {model_name})")
        
        retries = 0
        last_stdout = ""
        while retries < MAX_RETRIES:
            try:
                result = subprocess.run(
                    [sys.executable, str(code_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                last_stdout = result.stdout or ""
                if last_stdout:
                    print(last_stdout)
                print(f"[INFO] {code_path.name} executed successfully.\n")
                break
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr or ""
                stdout = exc.stdout or ""
                last_stdout = stdout
                detailed_error = "\n".join(
                    [
                        "STDERR:",
                        stderr.strip(),
                        "\nSTDOUT:",
                        stdout.strip(),
                        "\nProcess message:",
                        str(exc),
                    ]
                )
                retries += 1
                print(f"[ERROR] {code_path.name} execution failed. Attempting to fix and retry ({retries}/{MAX_RETRIES})...")

                code_content = code_path.read_text(encoding="utf-8")
                print(f"[INFO] Sending failing code to LLM for repair...")
                corrected_code = fix_code_with_llm(
                    code_content, 
                    detailed_error, 
                    model_recommendations, 
                    dataset_context,
                    dataset_report
                )
                code_path.write_text(corrected_code, encoding="utf-8")
                print(f"[INFO] Code fixed. Retrying execution...\n")
        
        # If succeeded, parse metrics from last_stdout
        if retries < MAX_RETRIES and last_stdout:
            mae_match = re.search(r"MAE[:\s]+([0-9eE+\-.]+)", last_stdout)
            mse_match = re.search(r"MSE[:\s]+([0-9eE+\-.]+)", last_stdout)
            r2_match = re.search(r"R2[:\s]+([0-9eE+\-.]+)", last_stdout)
            metrics = {}
            if mae_match:
                try:
                    metrics["MAE"] = float(mae_match.group(1))
                except ValueError:
                    metrics["MAE"] = mae_match.group(1)
            if mse_match:
                try:
                    metrics["MSE"] = float(mse_match.group(1))
                except ValueError:
                    metrics["MSE"] = mse_match.group(1)
            if r2_match:
                try:
                    metrics["R2"] = float(r2_match.group(1))
                except ValueError:
                    metrics["R2"] = r2_match.group(1)
            
            validation_metrics[model_name] = {
                "code_path": str(code_path.resolve()),
                "metrics": metrics,
            }
        
        if retries >= MAX_RETRIES:
            print(f"[ERROR] Maximum retries reached for {code_path.name}. "
                  "Please inspect the code manually.\n")
    
    # Persist metrics back to orchestrator_memory.json
    memory_data["validation_metrics"] = validation_metrics
    with memory_file.open("w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2)
    
    print("[INFO] All validation/testing code execution completed and metrics saved to orchestrator_memory.json.")


if __name__ == "__main__":
    main()

