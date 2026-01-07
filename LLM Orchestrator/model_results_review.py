import json
import os
from pathlib import Path

def load_memory():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    memory_file = script_dir / "orchestrator_memory.json"
    
    if not memory_file.exists():
        print(f"[ERROR] orchestrator_memory.json not found at: {memory_file}")
        return None
    with open(memory_file, "r", encoding="utf-8") as f:
        return json.load(f)


def print_model_metrics(validation_dict):
    print("\n================= MODEL PERFORMANCE METRICS =================\n")

    for model_name, model_info in validation_dict.items():
        metrics = model_info.get("metrics", {})
        print(f"Model: {model_name}")
        print(f"  MAE: {metrics.get('MAE')}")
        print(f"  MSE: {metrics.get('MSE')}")
        print(f"  R2:  {metrics.get('R2')}")
        print(f"  Code Path: {model_info.get('code_path')}")
        print()

    print("==============================================================\n")


def ask_yes_no(prompt):
    while True:
        choice = input(prompt + " (y/n): ").strip().lower()
        if choice in ["y", "n"]:
            return choice
        print("Invalid input. Enter 'y' or 'n'.")


def choose_model(validation_dict):
    models = list(validation_dict.keys())

    print("\nSelect the model you want to train on the FULL dataset:")
    for idx, model_name in enumerate(models, start=1):
        print(f"{idx}. {model_name}")

    while True:
        choice = input("Enter the number of the model: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            selected = models[int(choice) - 1]
            print(f"\n[INFO] You selected: {selected}")
            
            # Ask if user wants to generate full training code
            generate_code = ask_yes_no("Do you want to generate full training code for this model?")
            if generate_code == "y":
                return {"action": "generate_full_training", "model": selected}
            else:
                print("[INFO] Full training code generation skipped.\n")
                return {"action": "skip", "model": selected}
        print("Invalid choice. Try again.")


def improvement_menu():
    print("\nWhat would you like to do next?")
    print("1. Run improvement agent")
    print("2. Reselect models")
    print("3. Exit")

    while True:
        option = input("Enter option: ").strip()
        if option == "1":
            # Load master_memory.json
            master_memory = load_master_memory()
            if not master_memory:
                return
            
            # Get the list of models from selected_models_for_validation
            if "llm_orchestrator" not in master_memory:
                master_memory["llm_orchestrator"] = {}
            if "orchestrator_memory" not in master_memory["llm_orchestrator"]:
                master_memory["llm_orchestrator"]["orchestrator_memory"] = {}
            
            orchestrator_memory = master_memory["llm_orchestrator"]["orchestrator_memory"]
            models_list = orchestrator_memory.get("selected_models_for_validation", [])
            
            if not models_list:
                print("[ERROR] No models found in selected_models_for_validation.")
                return
            
            # Display models and let user select
            print("\nSelect the model you would like to improve:")
            for idx, model_name in enumerate(models_list, start=1):
                print(f"{idx}. {model_name}")
            
            while True:
                choice = input("Enter the number of the model: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(models_list):
                    selected_model = models_list[int(choice) - 1]
                    print(f"\n[INFO] You selected: {selected_model}")
                    
                    # Update master_memory.json
                    orchestrator_memory["selected_model_for_full_training"] = selected_model
                    save_master_memory(master_memory)
                    print(f"[INFO] Selected model '{selected_model}' saved to master_memory.json")
                    
                    # Also update orchestrator_memory.json to keep them in sync
                    orchestrator_memory_local = load_memory()
                    if orchestrator_memory_local:
                        orchestrator_memory_local["selected_model_for_full_training"] = selected_model
                        save_memory(orchestrator_memory_local)
                        print(f"[INFO] Selected model '{selected_model}' saved to orchestrator_memory.json")
                    
                    print("\n[INFO] Improvement Agent will run here.\n")
                    return
                print("Invalid choice. Try again.")
        elif option == "2":
            print("\n[INFO] (Placeholder) Model reselection will run here.\n")
            return
        elif option == "3":
            print("\n[INFO] Exiting program.\n")
            exit(0)
        else:
            print("Invalid input. Try again.")


def load_master_memory():
    """Load master_memory.json from project root"""
    script_dir = Path(__file__).parent
    master_memory_file = script_dir.parent / "master_memory.json"
    
    if not master_memory_file.exists():
        print(f"[ERROR] master_memory.json not found at: {master_memory_file}")
        return None
    with open(master_memory_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_master_memory(master_memory_data):
    """Save memory back to master_memory.json"""
    script_dir = Path(__file__).parent
    master_memory_file = script_dir.parent / "master_memory.json"
    with open(master_memory_file, "w", encoding="utf-8") as f:
        json.dump(master_memory_data, f, indent=2)


def save_memory(memory_data):
    """Save memory back to orchestrator_memory.json"""
    script_dir = Path(__file__).parent
    memory_file = script_dir / "orchestrator_memory.json"
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory_data, f, indent=2)


def main():
    memory = load_memory()
    if not memory:
        return None

    validation_dict = memory.get("validation_metrics", {})
    if not validation_dict:
        print("[ERROR] No validation metrics found inside orchestrator_memory.json.")
        return None

    # Print all model metrics
    print_model_metrics(validation_dict)

    # Ask if user is satisfied
    satisfied = ask_yes_no("Are you satisfied with these results?")
    if satisfied == "y":
        result = choose_model(validation_dict)
        # Save the selected model directly to orchestrator_memory.json
        if result and result.get("action") == "generate_full_training":
            selected_model = result.get("model")
            memory["selected_model_for_full_training"] = selected_model
            save_memory(memory)
            print(f"[INFO] Selected model '{selected_model}' saved to orchestrator_memory.json")
            return result
        elif result:
            # User selected a model but chose not to generate full training code
            # Still save the selection (optional, or clear it)
            memory["selected_model_for_full_training"] = None
            save_memory(memory)
        return result
    else:
        improvement_menu()
        return None


if __name__ == "__main__":
    main()
