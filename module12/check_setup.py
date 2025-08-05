#!/usr/bin/env python3
"""
Setup verification script for Streamlit deployment
"""
import os
import sys

def check_setup():
    print("=== Streamlit App Setup Verification ===")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check required files
    required_files = [
        'streamlit_app.py',
        'iris_model.pkl', 
        'model_info.pkl',
        'requirements.txt'
    ]
    
    print("\n=== File Check ===")
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} - NOT FOUND")
    
    # Check dependencies
    print("\n=== Dependency Check ===")
    try:
        import streamlit
        print(f"✅ streamlit {streamlit.__version__}")
    except ImportError:
        print("❌ streamlit - NOT INSTALLED")
    
    try:
        import pandas
        print(f"✅ pandas {pandas.__version__}")
    except ImportError:
        print("❌ pandas - NOT INSTALLED")
        
    try:
        import sklearn
        print(f"✅ scikit-learn {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn - NOT INSTALLED")
        
    try:
        import joblib
        print(f"✅ joblib {joblib.__version__}")
    except ImportError:
        print("❌ joblib - NOT INSTALLED")
        
    try:
        import numpy
        print(f"✅ numpy {numpy.__version__}")
    except ImportError:
        print("❌ numpy - NOT INSTALLED")
    
    # Test model loading
    print("\n=== Model Loading Test ===")
    try:
        import joblib
        model = joblib.load('iris_model.pkl')
        model_info = joblib.load('model_info.pkl')
        print("✅ Models loaded successfully")
        print(f"✅ Model type: {type(model)}")
        print(f"✅ Target classes: {model_info['target_names']}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")

if __name__ == "__main__":
    check_setup()
