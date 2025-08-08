import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="wide"
)

# Create model if it doesn't exist
@st.cache_resource
def create_model_if_needed():
    # Check if model files exist
    if not os.path.exists("iris_model.pkl") or not os.path.exists("model_info.pkl"):
        st.info("🤖 Creating ML model for the first time... This will take a moment!")

        with st.spinner("Training the machine learning model..."):
            # Load iris dataset
            iris = load_iris()

            # Convert to pandas DataFrame
            data = pd.DataFrame(iris.data, columns=iris.feature_names)
            target = pd.Series(iris.target)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                data, target, test_size=0.2, random_state=42
            )

            # Train model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Calculate accuracy
            accuracy = model.score(X_test, y_test)

            # Save model
            joblib.dump(model, "iris_model.pkl")

            # Save model info
            model_info = {
                "feature_names": iris.feature_names,
                "target_names": iris.target_names,
                "accuracy": accuracy,
            }
            joblib.dump(model_info, "model_info.pkl")

        st.success("✅ Model trained successfully! Refreshing the app...")
        st.rerun()


# Load the trained model and model information
@st.cache_resource
def load_model():
    model = joblib.load('iris_model.pkl')
    model_info = joblib.load('model_info.pkl')
    return model, model_info

# Main app
def main():
    st.title("🌸 Iris Flower Classifier")
    st.write("This app predicts the type of iris flower based on its measurements!")

    # Create model if needed (first time setup)
    create_model_if_needed()

    # Load model
    model, model_info = load_model()
    
    # Display model information
    st.sidebar.header("Model Information")
    st.sidebar.write(f"Model Accuracy: {model_info['accuracy']*100:.2f}%")
    st.sidebar.write("Features used:")
    for feature in model_info['feature_names']:
        st.sidebar.write(f"• {feature}")
    
    # Create input form
    st.header("Enter Flower Measurements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.4,
            step=0.1
        )
        
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=2.0,
            max_value=5.0,
            value=3.4,
            step=0.1
        )
    
    with col2:
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=1.0,
            max_value=7.0,
            value=1.3,
            step=0.1
        )
        
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=3.0,
            value=0.2,
            step=0.1
        )
    
    # Create prediction button
    if st.button("🔮 Predict Flower Type", type="primary"):
        # Prepare input data
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.header("Prediction Results")
        
        predicted_class = model_info['target_names'][prediction]
        st.success(f"Predicted Flower Type: **{predicted_class.title()}**")

        # Display flower result
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Try to show image from resources folder, fallback if not available
            image_paths = ["resources/image1b.png", "image1b.png"]
            image_displayed = False
            
            for image_path in image_paths:
                if os.path.exists(image_path):
                    try:
                        if predicted_class.lower() == "setosa":
                            st.image(image_path, caption=f"🌸 {predicted_class.title()} Iris Flower", use_container_width=True)
                        elif predicted_class.lower() == "versicolor":
                            st.image(image_path, caption=f"🌺 {predicted_class.title()} Iris Flower", use_container_width=True)
                        elif predicted_class.lower() == "virginica":
                            st.image(image_path, caption=f"🌷 {predicted_class.title()} Iris Flower", use_container_width=True)
                        else:
                            st.image(image_path, caption=f"🌻 {predicted_class.title()} Iris Flower", use_container_width=True)
                        image_displayed = True
                        break
                    except Exception:
                        continue
            
            # If no image could be displayed, just show the flower type
            if not image_displayed:
                st.info(f"🌸 Predicted: {predicted_class.title()} Iris Flower")

        # Show prediction probabilities
        st.subheader("Prediction Confidence")
        prob_df = pd.DataFrame({
            'Flower Type': model_info['target_names'],
            'Probability': prediction_proba
        })
        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(prob_df, use_container_width=True)
        
        # Show input summary
        st.subheader("Your Input")
        input_df = pd.DataFrame({
            'Measurement': model_info['feature_names'],
            'Value': [sepal_length, sepal_width, petal_length, petal_width]
        })
        st.dataframe(input_df, use_container_width=True)
    
    # Add information about iris types
    st.header("About Iris Flowers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🌸 Setosa")
        st.write("• Small petals")
        st.write("• Wide sepals")
        st.write("• Most distinct type")
    
    with col2:
        st.subheader("🌺 Versicolor")
        st.write("• Medium-sized petals")
        st.write("• Moderate sepal width")
        st.write("• Intermediate features")
    
    with col3:
        st.subheader("🌷 Virginica")
        st.write("• Large petals")
        st.write("• Long sepals")
        st.write("• Largest overall size")

if __name__ == "__main__":
    main()
