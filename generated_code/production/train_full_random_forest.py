import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def main():
    try:
        # Define absolute paths
        dataset_path = r"C:\Users\Danish\Desktop\llm-orchestration-agent\examples\AAPL.csv"
        model_save_path = r"C:\Users\Danish\Desktop\model_random_forest.pkl"
        
        # Step 1: Load the full dataset
        print("Loading dataset...")
        data = pd.read_csv(dataset_path, parse_dates=['Date'])
        
        # Extract day of week as target
        data['Day'] = data['Date'].dt.dayofweek
        
        # Ensure all non-date columns are numeric for RandomForestClassifier
        X = data.drop(columns=["Date", "Day"]).select_dtypes(include='number')
        y = data["Day"]
        
        # Print dataset shape for verification
        print(f"Dataset loaded with {data.shape[0]} samples and {X.shape[1]} features.")
        
        # Step 3: Train the model on the entire dataset
        print("Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=100,          # Number of trees in the forest
            random_state=42,           # For reproducibility
            class_weight='balanced',   # Handle imbalanced data
            max_depth=None,            # No limit on depth
            min_samples_split=2        # Minimum number of samples required to split an internal node
        )
        
        rf_model.fit(X, y)
        print("Model training completed.")
        
        # Step 4: Save the trained model using pickle
        with open(model_save_path, 'wb') as file:
            pickle.dump(rf_model, file)
        
        print(f"Model saved successfully at {model_save_path}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

# Example of how to load and use the model for prediction (as comments):
# import pickle
# with open(r"C:\Users\Danish\Desktop\model_random_forest.pkl", 'rb') as file:
#     loaded_model = pickle.load(file)
# 
# # Assuming `new_data` is a pandas DataFrame containing new samples
# predictions = loaded_model.predict(new_data)