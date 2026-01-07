import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv(r"C:\Users\Administrator\OneDrive - Higher Education Commission\Desktop\IntelliModel\backend\Pre Processing Agent\processed_output.csv")

# Check if all expected columns are present in the dataset
expected_columns = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
    'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
    'parking', 'prefarea', 'furnishingstatus', 'price'
]

missing_columns = [col for col in expected_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

# Split into features and target
X = data.drop('price', axis=1)
y = data['price']

# Define categorical columns based on dataset context
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                       'airconditioning', 'prefarea', 'furnishingstatus']

# Ensure all expected categorical columns are present in the features
missing_categorical_columns = [col for col in categorical_columns if col not in X.columns]
if missing_categorical_columns:
    raise ValueError(f"Missing expected categorical columns: {missing_categorical_columns}")

# Encode categorical columns
for column in categorical_columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate on test split
y_pred = model.predict(X_test)

# Print task-appropriate metrics
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))