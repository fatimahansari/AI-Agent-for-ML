import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

def main():
    try:
        # Define paths
        dataset_path = r"C:\Users\Danish\Desktop\llm-orchestration-agent\examples\AAPL.csv"
        model_save_path = r"C:\Users\Danish\Desktop\model_deployment.pkl"
        scaler_save_path = r"C:\Users\Danish\Desktop\scaler_deployment.pkl"

        # Load dataset
        print("Loading dataset...")
        data = pd.read_csv(dataset_path)

        # Convert date columns to datetime if necessary and extract features
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Extract features and target
        X = data.drop(columns=["Day", "Date"])
        y = data["Day"]
        
        # Print dataset shape for verification
        print(f"Dataset loaded with shape: {data.shape}")
        print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

        # Ensure X is numeric
        X = pd.get_dummies(X, drop_first=True)

        # Feature scaling if required
        scaler = None
        if True:  # Assuming we're using Logistic Regression which requires scaling
            print("Applying feature scaling...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        # Model training
        print("Training model on entire dataset...")
        model = LogisticRegression(
            max_iter=2000,
            multi_class='multinomial',
            solver='lbfgs',  # Suitable for multinomial logistic regression
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_scaled, y)

        # Save the model
        print(f"Saving model to {model_save_path}...")
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save the scaler if used
        if scaler is not None:
            print(f"Saving scaler to {scaler_save_path}...")
            with open(scaler_save_path, 'wb') as f:
                pickle.dump(scaler, f)

        # Confirmation messages
        print("Model and scaler (if applicable) saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example of how to load and use the model for prediction
def load_and_predict(sample_data):
    try:
        with open(r"C:\Users\Danish\Desktop\model_deployment.pkl", 'rb') as f:
            model = pickle.load(f)
        
        scaler_path = r"C:\Users\Danish\Desktop\scaler_deployment.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            sample_data_scaled = scaler.transform(sample_data)
        else:
            sample_data_scaled = sample_data

        prediction = model.predict(sample_data_scaled)
        return prediction
    except Exception as e:
        print(f"An error occurred during loading or predicting: {e}")

if __name__ == "__main__":
    main()