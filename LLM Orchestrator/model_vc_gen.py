import json
from pathlib import Path
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DEFAULT_MODEL = "phi4"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"

# Set output folder inside the LLM Orchestrator directory
current_dir = Path(__file__).parent
OUTPUT_FOLDER = current_dir / "model_testing"
OUTPUT_FOLDER.mkdir(exist_ok=True)


def _strip_code_fences(code: str) -> str:
    """Remove surrounding markdown code fences if present."""
    if not code:
        return code
    cleaned = code.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        newline_idx = cleaned.find("\n")
        if newline_idx != -1:
            first_line = cleaned[:newline_idx].strip().lower()
            # Drop optional language hint (e.g., python)
            if first_line.isalpha():
                cleaned = cleaned[newline_idx + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def build_prompt():
    return ChatPromptTemplate.from_template(
        """
You are an ML testing code generator.
Base everything ONLY on the dataset context, dataset report, and the recommended ML model below.
Do NOT invent columns or metadata. Use ONLY what is in the report.

Task:
Write Python code that:
- Loads the cleaned dataset from the exact path: {dataset_file}
- Splits into train/test
- Encodes categorical columns if needed (based on report)
- Trains the given ML model
- Evaluates on test split
- Prints task-appropriate metrics:
    * Regression → MAE, MSE, R2
    * Classification → accuracy, precision, recall

STRICT RULES:
- OUTPUT ONLY PYTHON CODE
- NO COMMENTS
- NO MARKDOWN
- NO EXPLANATIONS
- Use the exact full path provided: {dataset_file}
- The path is absolute and should be used as-is in pd.read_csv()

Model to implement:
{model}

Dataset Context:
{context}

Dataset Report:
{report}
"""
    )


def generate_test_code(model_name, context, report, dataset_file, model=DEFAULT_MODEL, endpoint=DEFAULT_ENDPOINT):
    llm = ChatOllama(
        model=model,
        base_url=endpoint,
        temperature=0.1
    )

    prompt = build_prompt()
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "model": model_name,
        "context": context,
        "report": report,
        "dataset_file": dataset_file
    })


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate testing code for model recommendations")
    parser.add_argument("--memory", type=Path, default=None,
                        help="Path to orchestrator_memory.json (default: LLM Orchestrator directory/orchestrator_memory.json)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="LLM model name")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT,
                        help="LLM endpoint URL")
    args = parser.parse_args()

    # Resolve memory file path
    if args.memory is None:
        # Default to orchestrator_memory.json in the LLM Orchestrator directory
        memory_path = current_dir / "orchestrator_memory.json"
    else:
        memory_path = Path(args.memory)
        # If relative path, try current directory first, then script directory
        if not memory_path.is_absolute():
            if not memory_path.exists():
                # Try in script directory
                alt_path = current_dir / memory_path
                if alt_path.exists():
                    memory_path = alt_path
                else:
                    # Try in current working directory
                    cwd_path = Path.cwd() / memory_path
                    if cwd_path.exists():
                        memory_path = cwd_path

    # ------ LOAD orchestrator_memory.json ------
    if not memory_path.exists():
        raise FileNotFoundError(f"orchestrator_memory.json not found at: {memory_path}")

    with memory_path.open("r", encoding="utf-8") as f:
        orchestrator_memory = json.load(f)

    # Get master_memory_path from orchestrator_memory
    master_memory_path = orchestrator_memory.get("master_memory_path")
    if not master_memory_path:
        raise ValueError("'master_memory_path' not found in orchestrator_memory.json")

    master_memory_path = Path(master_memory_path)
    if not master_memory_path.exists():
        raise FileNotFoundError(f"Master memory file not found: {master_memory_path}")

    # Load master_memory.json
    with master_memory_path.open("r", encoding="utf-8") as f:
        master_memory = json.load(f)

    # Get dataset path from master_memory.json
    # Prefer processed_output_path, fallback to dataset_path
    # Use full path, not just filename
    processed_output_path = master_memory.get("preprocessing", {}).get("processed_output_path", "")
    dataset_path = master_memory.get("dataset_path", "")
    
    if processed_output_path:
        # Use full absolute path
        dataset_file_path = Path(processed_output_path)
        if not dataset_file_path.is_absolute():
            # If relative, resolve relative to master_memory.json location
            master_memory_dir = master_memory_path.parent
            dataset_file_path = (master_memory_dir / dataset_file_path).resolve()
        dataset_file = str(dataset_file_path)
    elif dataset_path:
        # Use full absolute path
        dataset_file_path = Path(dataset_path)
        if not dataset_file_path.is_absolute():
            # If relative, resolve relative to master_memory.json location
            master_memory_dir = master_memory_path.parent
            dataset_file_path = (master_memory_dir / dataset_file_path).resolve()
        dataset_file = str(dataset_file_path)
    else:
        raise ValueError("Neither 'processed_output_path' nor 'dataset_path' found in master_memory.json")

    # Get report from master_memory.json
    report_json = master_memory.get("preprocessing", {}).get("report", {})
    if not report_json:
        raise ValueError("'report' not found in master_memory.json under 'preprocessing'")

    # Get selected models for validation from orchestrator_memory.json
    selected_models_for_validation = orchestrator_memory.get("selected_models_for_validation", [])
    
    if not selected_models_for_validation:
        raise ValueError(
            "'selected_models_for_validation' not found in orchestrator_memory.json. "
            "Please select models for validation first."
        )
    
    # Get structured model recommendations from orchestrator_memory.json
    structured_recommendations = orchestrator_memory.get("model_recommendations_structured", [])
    context = orchestrator_memory.get("context_of_dataset", "")

    if not structured_recommendations:
        # Fallback: try to parse raw recommendations if structured format not available
        raw_recommendations = orchestrator_memory.get("model_recommendations", "")
        if raw_recommendations:
            print("Warning: Using raw recommendations, structured format not found. Parsing...")
            # Simple extraction from raw text
            import re
            model_lines = []
            for line in raw_recommendations.splitlines():
                # Look for numbered list items with model names
                match = re.search(r'\d+\.\s*\*\*?(.+?)\*\*?', line)
                if match:
                    model_lines.append(match.group(1).strip())
            
            if model_lines:
                structured_recommendations = [
                    {"Model Recommended": model, "Reason": "No reason provided"}
                    for model in model_lines
                ]
            else:
                raise ValueError("No model recommendations found in orchestrator_memory.json")
        else:
            raise ValueError("'model_recommendations_structured' not found in orchestrator_memory.json")

    if not context:
        raise ValueError("'context_of_dataset' missing in orchestrator_memory.json")

    report_str = json.dumps(report_json, indent=2)

    print("\n=== Generating validation/testing code for selected models ===\n")
    print(f"Selected models for validation: {', '.join(selected_models_for_validation)}\n")

    generated_files = []
    
    # Filter structured recommendations to only include selected models
    filtered_recommendations = [
        entry for entry in structured_recommendations
        if entry.get("Model Recommended", "") in selected_models_for_validation
    ]
    
    # Use filtered recommendations (only selected models)
    for model_entry in filtered_recommendations:
        model_name = model_entry.get("Model Recommended", "")
        if not model_name:
            continue
        # Use plain ASCII arrow to avoid UnicodeEncodeError on some consoles
        print(f"-> Generating testing code for: {model_name}")

        code = generate_test_code(
            model_name=model_name,
            context=context,
            report=report_str,
            dataset_file=dataset_file,
            model=args.model,
            endpoint=args.endpoint
        )

        # Strip code fences if present
        code = _strip_code_fences(code)

        # Save output file
        file_name = model_name.lower().replace(" ", "_") + "_test.py"
        file_path = OUTPUT_FOLDER / file_name

        with file_path.open("w", encoding="utf-8") as f:
            f.write(code)

        generated_files.append(str(file_path.resolve()))
        print(f"Saved: {file_path}")

    print("\n=== DONE! Testing code saved in 'model_testing/' folder ===")
    
    return generated_files


if __name__ == "__main__":
    main()
