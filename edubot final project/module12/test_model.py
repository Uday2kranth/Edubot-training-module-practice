# Test script to verify model functionality
import joblib

# Load the model and model info
model = joblib.load('iris_model.pkl')
model_info = joblib.load('model_info.pkl')

# Test with sample data
test_samples = [
    [5.1, 3.5, 1.4, 0.2],  # Should be Setosa
    [6.0, 3.0, 4.5, 1.5],  # Should be Versicolor  
    [6.5, 3.0, 5.5, 2.0]   # Should be Virginica
]

print("Testing the saved model:")
print("=" * 40)

for i, sample in enumerate(test_samples):
    prediction = model.predict([sample])
    probabilities = model.predict_proba([sample])
    predicted_species = model_info['target_names'][prediction[0]]
    
    print(f"Test {i+1}:")
    print(f"Input: {sample}")
    print(f"Predicted Species: {predicted_species}")
    print(f"Probabilities: {dict(zip(model_info['target_names'], probabilities[0]))}")
    print("-" * 30)

print("Model testing completed successfully!")
