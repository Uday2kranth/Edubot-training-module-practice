
import streamlit as st
import joblib
import numpy as np

# Load the model and model info
@st.cache_resource
def load_model():
    model = joblib.load('iris_model.pkl')
    model_info = joblib.load('model_info.pkl')
    return model, model_info

# Load model
model, model_info = load_model()

# App title
st.title('Iris Flower Classifier')

# App description
st.write('This app predicts the species of iris flowers based on their measurements.')

# Create input widgets
st.sidebar.header('Input Features')

sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.0)

# Create input array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Display input values
st.subheader('Input Values')
st.write('Sepal Length:', sepal_length)
st.write('Sepal Width:', sepal_width)
st.write('Petal Length:', petal_length)
st.write('Petal Width:', petal_width)

# Make prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display prediction
st.subheader('Prediction')
predicted_species = model_info['target_names'][prediction[0]]
st.write('Predicted Species:', predicted_species)

# Display prediction probabilities
st.subheader('Prediction Probabilities')
for i, species in enumerate(model_info['target_names']):
    st.write(f'{species}: {prediction_proba[0][i]:.3f}')
