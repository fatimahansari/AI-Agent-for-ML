"""Example script showing how to generate metadata from a dataset."""

from pathlib import Path

from src.metadata_generator import generate_metadata_from_file

# Example 1: Basic usage - auto-detect everything
if __name__ == "__main__":
    # Generate metadata for AAPL dataset
    data_path = Path("AAPL.csv")
    
    metadata, output_path = generate_metadata_from_file(
        data_path=data_path,
        output_path=Path("AAPL_metadata.json"),
        target="Close_forcast",  # Specify target column
        problem_type="regression",  # Specify problem type
        domain="finance",  # Specify domain
        constraints=[
            "Temporal ordering must be preserved - use time-based cross-validation",
            "Predictions should account for market volatility",
        ],
        evaluation_metric="rmse",
        notes="Stock price forecasting dataset with technical indicators and market indices.",
    )
    
    print(f"\nGenerated metadata saved to: {output_path}")
    print(f"\nMetadata summary:")
    print(f"  Name: {metadata.name}")
    print(f"  Problem type: {metadata.problem_type}")
    print(f"  Target: {metadata.target}")
    print(f"  Samples: {metadata.num_samples:,}")
    print(f"  Features: {metadata.num_features}")
    print(f"  Domain: {metadata.domain}")
    print(f"  Time index: {metadata.time_index}")

