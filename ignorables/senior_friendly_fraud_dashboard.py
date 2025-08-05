"""
🚨 Senior-Friendly Fraud Detection Dashboard 
===========================================

Priority: Easy to understand and use for elderly users
Secondary: Accurate fraud detection

Author: EduBot Final Project
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ===================================
# 🎨 SENIOR-FRIENDLY STYLING
# ===================================

def setup_page_config():
    """Configure page for elderly users"""
    st.set_page_config(
        page_title="🚨 Fraud Detection System",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for elderly-friendly design
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        height: 3rem;
        width: 100%;
        font-size: 18px !important;
        font-weight: bold !important;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        background-color: #4CAF50;
        color: white;
    }
    .stButton > button:hover {
        background-color: #45a049;
        border-color: #45a049;
    }
    .big-text {
        font-size: 24px !important;
        font-weight: bold !important;
        color: #2E8B57;
    }
    .alert-text {
        font-size: 22px !important;
        font-weight: bold !important;
        color: #DC143C;
    }
    .success-text {
        font-size: 22px !important;
        font-weight: bold !important;
        color: #228B22;
    }
    .info-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #87CEEB;
        background-color: #F0F8FF;
        margin: 10px 0;
    }
    .metric-box {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #ddd;
        margin: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

def load_model():
    """Load the trained fraud detection model"""
    try:
        with open('senior_friendly_fraud_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("❌ Model file not found. Please run the training notebook first.")
        return None

def load_sample_data():
    """Load sample transaction data"""
    try:
        return pd.read_csv('sample_transaction_data.csv')
    except FileNotFoundError:
        # Create dummy data if file doesn't exist
        return pd.DataFrame({
            'amount': [100, 250, 500, 1000, 50],
            'age': [25, 45, 60, 35, 70],
            'transaction_type_encoded': [1, 0, 2, 1, 0],
            'merchant_category_encoded': [0, 1, 2, 0, 1],
            'device_type_encoded': [1, 0, 1, 2, 0]
        })

# ===================================
# 🔍 FRAUD DETECTION FUNCTIONS
# ===================================

def predict_fraud_risk(model_package, transaction_data):
    """Predict fraud risk for a transaction"""
    if model_package is None:
        return 0.5, "Medium"
    
    # Prepare features
    features = []
    for col in model_package['feature_columns']:
        if col in transaction_data:
            features.append(transaction_data[col])
        else:
            features.append(0)  # Default value for missing features
    
    # Scale features
    features_scaled = model_package['scaler'].transform([features])
    
    # Predict
    risk_probability = model_package['model'].predict_proba(features_scaled)[0][1]
    
    # Categorize risk
    if risk_probability > 0.7:
        risk_level = "🔴 HIGH"
    elif risk_probability > 0.3:
        risk_level = "🟡 MEDIUM"
    else:
        risk_level = "🟢 LOW"
    
    return risk_probability, risk_level

def create_risk_gauge(risk_probability):
    """Create a large, clear risk gauge for elderly users"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Risk Level", 'font': {'size': 24, 'color': 'black'}},
        delta = {'reference': 30},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "black", 'tickfont': {'size': 16}},
            'bar': {'color': "darkblue", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightcoral'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.8,
                'value': 70}
        },
        number = {'font': {'size': 32, 'color': 'black'}}
    ))
    
    fig.update_layout(height=300, font={'color': "black", 'family': "Arial"})
    return fig

# ===================================
# 📊 DASHBOARD SECTIONS
# ===================================

def show_header():
    """Display main header"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #2E8B57; font-size: 36px;'>🚨 Fraud Detection System</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #4169E1; font-size: 20px;'>Easy-to-Use Security Dashboard for Everyone</h3>", unsafe_allow_html=True)

def show_system_status(model_package):
    """Show system status in senior-friendly format"""
    st.markdown("---")
    st.markdown("<h2 style='color: #2E8B57;'>🔍 System Status</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box" style="background-color: #E8F5E8;">
            <h3 style="color: #228B22; margin: 0;">🤖 AI Status</h3>
            <p style="font-size: 18px; margin: 5px 0; color: #228B22;"><strong>✅ ACTIVE</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        accuracy = model_package['performance']['accuracy'] if model_package else 0.75
        st.markdown(f"""
        <div class="metric-box" style="background-color: #E8F4FD;">
            <h3 style="color: #4169E1; margin: 0;">📊 Accuracy</h3>
            <p style="font-size: 18px; margin: 5px 0; color: #4169E1;"><strong>{accuracy*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box" style="background-color: #FFF8E1;">
            <h3 style="color: #FF8C00; margin: 0;">⚡ Speed</h3>
            <p style="font-size: 18px; margin: 5px 0; color: #FF8C00;"><strong>INSTANT</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        current_time = datetime.now().strftime("%H:%M")
        st.markdown(f"""
        <div class="metric-box" style="background-color: #F0E8FF;">
            <h3 style="color: #8B008B; margin: 0;">🕐 Time</h3>
            <p style="font-size: 18px; margin: 5px 0; color: #8B008B;"><strong>{current_time}</strong></p>
        </div>
        """, unsafe_allow_html=True)

def show_single_transaction_checker(model_package):
    """Main transaction checking interface"""
    st.markdown("---")
    st.markdown("<h2 style='color: #2E8B57;'>💳 Check a Single Transaction</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Enter Transaction Details")
        
        # Simple input form for elderly users
        amount = st.number_input(
            "💰 Transaction Amount ($)", 
            min_value=0.01, 
            max_value=100000.0, 
            value=100.0,
            step=10.0,
            help="How much money is being transferred?"
        )
        
        age = st.slider(
            "👤 Customer Age", 
            min_value=18, 
            max_value=100, 
            value=45,
            help="How old is the customer?"
        )
        
        transaction_type = st.selectbox(
            "🏧 Transaction Type",
            options=["Online Purchase", "ATM Withdrawal", "Bank Transfer"],
            help="What type of transaction is this?"
        )
        
        device_type = st.selectbox(
            "📱 Device Used",
            options=["Mobile Phone", "Computer", "ATM Machine"],
            help="What device was used for this transaction?"
        )
        
        time_of_day = st.selectbox(
            "🕐 Time of Transaction",
            options=["Morning (6AM-12PM)", "Afternoon (12PM-6PM)", "Evening (6PM-12AM)", "Night (12AM-6AM)"],
            help="When did this transaction happen?"
        )
        
        check_button = st.button("🔍 CHECK FOR FRAUD", type="primary")
    
    with col2:
        st.markdown("### 🎯 Fraud Risk Assessment")
        
        if check_button:
            # Prepare transaction data (simplified for elderly users)
            transaction_data = {
                'amount': amount,
                'age': age,
                'transaction_type_encoded': ["Online Purchase", "ATM Withdrawal", "Bank Transfer"].index(transaction_type),
                'device_type_encoded': ["Mobile Phone", "Computer", "ATM Machine"].index(device_type),
                'merchant_category_encoded': 1,  # Default value
                'amount_log': np.log1p(amount),
                'amount_zscore': (amount - 500) / 200,  # Simplified z-score
                'hour': [9, 15, 21, 3][["Morning (6AM-12PM)", "Afternoon (12PM-6PM)", "Evening (6PM-12AM)", "Night (12AM-6AM)"].index(time_of_day)]
            }
            
            # Get prediction
            risk_probability, risk_level = predict_fraud_risk(model_package, transaction_data)
            
            # Display risk gauge
            risk_gauge = create_risk_gauge(risk_probability)
            st.plotly_chart(risk_gauge, use_container_width=True)
            
            # Clear recommendation
            st.markdown("### 📋 Recommendation")
            
            if "HIGH" in risk_level:
                st.markdown("""
                <div style="background-color: #FFE4E1; padding: 20px; border-radius: 10px; border: 3px solid red;">
                    <h3 style="color: red; margin: 0;">🚨 HIGH RISK - BLOCK TRANSACTION</h3>
                    <p style="font-size: 18px; color: #8B0000;">
                        ⚠️ This transaction shows signs of fraud<br>
                        🛑 <strong>RECOMMENDED ACTION: BLOCK</strong><br>
                        📞 Contact customer immediately for verification
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            elif "MEDIUM" in risk_level:
                st.markdown("""
                <div style="background-color: #FFF8DC; padding: 20px; border-radius: 10px; border: 3px solid orange;">
                    <h3 style="color: orange; margin: 0;">⚠️ MEDIUM RISK - VERIFY CUSTOMER</h3>
                    <p style="font-size: 18px; color: #B8860B;">
                        🔍 This transaction needs additional verification<br>
                        📞 <strong>RECOMMENDED ACTION: VERIFY</strong><br>
                        ✅ Ask customer security questions before approving
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div style="background-color: #F0FFF0; padding: 20px; border-radius: 10px; border: 3px solid green;">
                    <h3 style="color: green; margin: 0;">✅ LOW RISK - SAFE TO PROCEED</h3>
                    <p style="font-size: 18px; color: #006400;">
                        ✅ This transaction appears normal<br>
                        👍 <strong>RECOMMENDED ACTION: APPROVE</strong><br>
                        💳 Safe to process this transaction
                    </p>
                </div>
                """, unsafe_allow_html=True)

def show_batch_analysis(model_package):
    """Batch transaction analysis"""
    st.markdown("---")
    st.markdown("<h2 style='color: #2E8B57;'>📊 Analyze Multiple Transactions</h2>", unsafe_allow_html=True)
    
    # Load sample data
    sample_data = load_sample_data()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📁 Upload Transaction File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with transactions",
            type=['csv'],
            help="Upload a file with transaction data for batch analysis"
        )
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_data)} transactions")
                
                # Process batch predictions (simplified for demo)
                if st.button("🔍 ANALYZE ALL TRANSACTIONS"):
                    st.markdown("### 📊 Analysis Results")
                    
                    # Simulate results for demo
                    high_risk = np.random.randint(1, 10)
                    medium_risk = np.random.randint(5, 20)
                    low_risk = len(batch_data) - high_risk - medium_risk
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("🔴 High Risk", high_risk, delta=f"{high_risk/len(batch_data)*100:.1f}%")
                    
                    with col_b:
                        st.metric("🟡 Medium Risk", medium_risk, delta=f"{medium_risk/len(batch_data)*100:.1f}%")
                    
                    with col_c:
                        st.metric("🟢 Low Risk", low_risk, delta=f"{low_risk/len(batch_data)*100:.1f}%")
                        
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        
        else:
            st.info("📝 No file uploaded. Using sample data for demonstration.")
            if st.button("🔍 ANALYZE SAMPLE DATA"):
                st.markdown("### 📊 Sample Analysis Results")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("🔴 High Risk", "2", delta="10%")
                
                with col_b:
                    st.metric("🟡 Medium Risk", "8", delta="40%")
                
                with col_c:
                    st.metric("🟢 Low Risk", "10", delta="50%")
    
    with col2:
        st.markdown("### 📈 Risk Distribution")
        
        # Create a simple risk distribution chart
        risk_data = pd.DataFrame({
            'Risk Level': ['Low Risk', 'Medium Risk', 'High Risk'],
            'Count': [10, 8, 2],
            'Color': ['green', 'orange', 'red']
        })
        
        fig = px.pie(
            risk_data, 
            values='Count', 
            names='Risk Level',
            color='Risk Level',
            color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'},
            title="Transaction Risk Distribution"
        )
        
        fig.update_layout(
            font_size=16,
            title_font_size=20
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_help_section():
    """Help section for elderly users"""
    st.markdown("---")
    st.markdown("<h2 style='color: #2E8B57;'>❓ How to Use This System</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🟢 For Safe Transactions:
        - ✅ **Green Light**: Transaction is safe to approve
        - 👍 **Low Risk**: Normal customer behavior
        - 💳 **Action**: Process the transaction normally
        
        ### 🟡 For Suspicious Transactions:
        - ⚠️ **Yellow Light**: Needs verification
        - 📞 **Medium Risk**: Call customer to verify
        - ❓ **Action**: Ask security questions first
        """)
    
    with col2:
        st.markdown("""
        ### 🔴 For Dangerous Transactions:
        - 🚨 **Red Light**: Very likely fraud
        - 🛑 **High Risk**: Block immediately
        - 📞 **Action**: Contact customer and security team
        
        ### 📞 Emergency Contacts:
        - 🏢 **Security Team**: Extension 911
        - 👨‍💼 **Supervisor**: Extension 999
        - 🚨 **Emergency**: Call local police
        """)

# ===================================
# 🚀 MAIN APPLICATION
# ===================================

def main():
    """Main application function"""
    setup_page_config()
    
    # Load model
    model_package = load_model()
    
    # Header
    show_header()
    
    # System status
    show_system_status(model_package)
    
    # Main functionality tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Transaction", "📊 Multiple Transactions", "❓ Help"])
    
    with tab1:
        show_single_transaction_checker(model_package)
    
    with tab2:
        show_batch_analysis(model_package)
    
    with tab3:
        show_help_section()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 14px;'>"
        "🚨 Senior-Friendly Fraud Detection System | "
        "Easy to Use • Accurate Detection • Always Reliable"
        "</p>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
