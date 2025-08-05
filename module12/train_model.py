# Simple ML Model Training Script
# This script trains a model to predict iris flower types
# No complex loops, conditions, or error handling as requested

import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Loading iris dataset...")
iris = load_iris()

# Convert to pandas DataFrame for easier handling
data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = pd.Series(iris.target)

print("Dataset loaded successfully!")
print(f"Dataset shape: {data.shape}")
print(f"Features: {list(data.columns)}")
print(f"Target classes: {iris.target_names}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

print("Training the model...")
# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the trained model
print("Saving the model...")
joblib.dump(model, 'iris_model.pkl')

# Save model information
model_info = {
    'feature_names': iris.feature_names,
    'target_names': iris.target_names,
    'accuracy': accuracy
}
joblib.dump(model_info, 'model_info.pkl')

print("Model saved successfully!")
print("Files created:")
print("- iris_model.pkl (the trained model)")
print("- model_info.pkl (model information)")
