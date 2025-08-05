@echo off
echo ========================================
echo Real-Time Fraud Monitoring Dashboard
echo ========================================
echo.
echo Starting the fraud detection dashboard...
echo Dashboard will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo ========================================
echo.

python -m streamlit run fraud_monitoring_dashboard.py

pause
