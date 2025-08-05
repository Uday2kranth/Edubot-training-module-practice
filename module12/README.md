# Iris Flower Classification Project

## Overview
This project builds a machine learning model to classify iris flowers into three species: setosa, versicolor, and virginica. The model uses flower measurements to make predictions.

## Features
- Machine learning model using Random Forest classifier
- Streamlit web application for easy predictions
- Model accuracy: 100% on test data

## Setup Instructions

1. Install Python 3.8 or higher
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## Input Examples
The model expects four measurements:
- Sepal Length: 4.0 - 8.0 cm
- Sepal Width: 2.0 - 4.5 cm  
- Petal Length: 1.0 - 7.0 cm
- Petal Width: 0.1 - 2.5 cm

## Sample Predictions
- Input: [5.1, 3.5, 1.4, 0.2] -> Prediction: Setosa
- Input: [6.0, 3.0, 4.5, 1.5] -> Prediction: Versicolor
- Input: [6.5, 3.0, 5.5, 2.0] -> Prediction: Virginica

## Files
- iris_model.pkl: Trained machine learning model
- model_info.pkl: Model metadata and class names
- streamlit_app.py: Web application
- requirements.txt: Required Python packages
- ml_model_deployment.ipynb: Complete development notebook

## Deployment
This app can be deployed on cloud platforms like:
- Streamlit Cloud
- Heroku
- AWS
- Google Cloud Platform

For Streamlit Cloud deployment:
1. Upload files to GitHub repository
2. Connect repository to Streamlit Cloud
3. Deploy with one click

### Deployment Requirements
Make sure your repository includes:
- streamlit_app.py (main app file)
- iris_model.pkl (trained model)
- model_info.pkl (model metadata)
- requirements.txt (dependencies)

### Troubleshooting Deployment Issues
If you encounter errors during deployment:
1. Check that all required files are in the repository
2. Ensure requirements.txt has the correct dependencies
3. Verify model files are not too large (< 100MB for Streamlit Cloud)
4. Check app logs for specific error messages
