"""
Real-Time Fraud Monitoring Dashboard Launcher

To run this dashboard:
1. Open terminal/command prompt
2. Navigate to the project folder
3. Run: streamlit run fraud_monitoring_dashboard.py

The dashboard will open in your web browser at http://localhost:8501
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages if not available"""
    required_packages = [
        'streamlit',
        'plotly',
        'scikit-learn',
        'xgboost',
        'pandas',
        'numpy'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    try:
        # Install requirements
        install_requirements()
        
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dashboard_path = os.path.join(script_dir, "fraud_monitoring_dashboard.py")
        
        # Launch streamlit
        print("Launching Real-Time Fraud Monitoring Dashboard...")
        print("Dashboard will open in your web browser at http://localhost:8501")
        print("Press Ctrl+C to stop the dashboard")
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])
        
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except Exception as e:
        print(f"Error launching dashboard: {e}")

if __name__ == "__main__":
    launch_dashboard()
