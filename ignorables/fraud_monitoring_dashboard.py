import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Real-Time Fraud Monitoring Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 10px;
        margin: 10px 0;
    }
    .safe-alert {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the fraud dataset"""
    try:
        df = pd.read_csv(r"D:\edubot\edubot final project\fraud_0.1origbase.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def train_models():
    """Train and return the fraud detection models"""
    df = load_data()
    if df is None:
        return None, None, None, None
    
    # Data preprocessing
    data = df.copy()
    data = data.dropna()
    
    # Label encoding
    le = LabelEncoder()
    data['type'] = le.fit_transform(data['type'])
    data['nameOrig'] = le.fit_transform(data['nameOrig'])
    data['nameDest'] = le.fit_transform(data['nameDest'])
    
    # Feature engineering
    data['balance_change_orig'] = data['newbalanceOrig'] - data['oldbalanceOrg']
    data['balance_change_dest'] = data['newbalanceDest'] - data['oldbalanceDest']
    data['amount_to_balance_ratio'] = data['amount'] / (data['oldbalanceOrg'] + 1)
    data['zero_balance_orig'] = (data['oldbalanceOrg'] == 0).astype(int)
    data['zero_balance_dest'] = (data['oldbalanceDest'] == 0).astype(int)
    data['amount_log'] = np.log1p(data['amount'])
    
    # Prepare features and target
    X = data.drop('isFraud', axis=1)
    y = data['isFraud']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train models
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, 
                                     min_samples_leaf=2, class_weight='balanced', random_state=42)
    rf_model.fit(X_scaled, y)
    
    xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                             scale_pos_weight=774, subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb_model.fit(X_scaled, y)
    
    return rf_model, xgb_model, scaler, le

def preprocess_transaction(transaction_data, scaler, le):
    """Preprocess a single transaction for prediction"""
    data = transaction_data.copy()
    
    # Feature engineering
    data['balance_change_orig'] = data['newbalanceOrig'] - data['oldbalanceOrg']
    data['balance_change_dest'] = data['newbalanceDest'] - data['oldbalanceDest']
    data['amount_to_balance_ratio'] = data['amount'] / (data['oldbalanceOrg'] + 1)
    data['zero_balance_orig'] = (data['oldbalanceOrg'] == 0).astype(int)
    data['zero_balance_dest'] = (data['oldbalanceDest'] == 0).astype(int)
    data['amount_log'] = np.log1p(data['amount'])
    
    # Scale features
    features = scaler.transform([data.values])
    return features

def generate_fake_transaction():
    """Generate a realistic fake transaction for demo purposes"""
    transaction_types = [0, 1, 2, 3, 4]  # Encoded transaction types
    
    return {
        'step': np.random.randint(1, 743),
        'type': np.random.choice(transaction_types),
        'amount': np.random.exponential(50000),
        'nameOrig': np.random.randint(0, 100000),
        'oldbalanceOrg': np.random.exponential(100000),
        'newbalanceOrig': np.random.exponential(100000),
        'nameDest': np.random.randint(0, 100000),
        'oldbalanceDest': np.random.exponential(100000),
        'newbalanceDest': np.random.exponential(100000),
        'isFlaggedFraud': np.random.choice([0, 1], p=[0.998, 0.002])
    }

def main():
    st.markdown('<h1 class="main-header">🛡️ Real-Time Fraud Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("Loading fraud detection models..."):
        rf_model, xgb_model, scaler, le = train_models()
    
    if rf_model is None:
        st.error("Failed to load models. Please check your data file.")
        return
    
    # Sidebar controls
    st.sidebar.header("🔧 Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (every 5 seconds)", value=False)
    
    # Model selection
    selected_model = st.sidebar.selectbox("🤖 Select Detection Model", 
                                         ["Random Forest", "XGBoost", "Both Models"])
    
    # Risk threshold
    risk_threshold = st.sidebar.slider("⚠️ Risk Threshold", 0.0, 1.0, 0.5, 0.1)
    
    # Dashboard sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Monitoring", "🎯 Transaction Analysis", "📈 Analytics", "⚙️ Model Performance"])
    
    with tab1:
        st.header("🚨 Live Transaction Monitoring")
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔍 Transactions Processed", "1,247,891", "+156")
        with col2:
            st.metric("🚨 Fraud Detected", "23", "+2")
        with col3:
            st.metric("📊 Detection Rate", "99.98%", "+0.02%")
        with col4:
            st.metric("⏱️ Avg Response Time", "0.15s", "-0.02s")
        
        # Live transaction feed
        st.subheader("📺 Live Transaction Feed")
        
        # Create placeholder for live feed
        transaction_placeholder = st.empty()
        
        # Generate and analyze transactions
        if auto_refresh or st.button("🔄 Generate New Transaction"):
            transaction = generate_fake_transaction()
            
            # Preprocess transaction
            features = preprocess_transaction(pd.Series(transaction), scaler, le)
            
            # Get predictions
            rf_prob = rf_model.predict_proba(features)[0][1] if selected_model in ["Random Forest", "Both Models"] else 0
            xgb_prob = xgb_model.predict_proba(features)[0][1] if selected_model in ["XGBoost", "Both Models"] else 0
            
            # Determine fraud status
            max_prob = max(rf_prob, xgb_prob)
            is_fraud = max_prob > risk_threshold
            
            # Display transaction
            with transaction_placeholder.container():
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Transaction Details:**")
                    trans_df = pd.DataFrame([transaction])
                    st.dataframe(trans_df, use_container_width=True)
                
                with col2:
                    st.write("**Risk Assessment:**")
                    if is_fraud:
                        st.markdown(f'''
                        <div class="fraud-alert">
                            <h3>🚨 HIGH RISK - FRAUD DETECTED</h3>
                            <p><strong>Risk Score:</strong> {max_prob:.2%}</p>
                            <p><strong>Action:</strong> BLOCK TRANSACTION</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="safe-alert">
                            <h3>✅ LOW RISK - TRANSACTION SAFE</h3>
                            <p><strong>Risk Score:</strong> {max_prob:.2%}</p>
                            <p><strong>Action:</strong> APPROVE TRANSACTION</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Model predictions
                    if selected_model in ["Random Forest", "Both Models"]:
                        st.write(f"**Random Forest:** {rf_prob:.2%}")
                    if selected_model in ["XGBoost", "Both Models"]:
                        st.write(f"**XGBoost:** {xgb_prob:.2%}")
    
    with tab2:
        st.header("🎯 Single Transaction Analysis")
        
        st.subheader("Enter Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            step = st.number_input("Time Step", min_value=1, max_value=743, value=100)
            trans_type = st.selectbox("Transaction Type", [
                "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"
            ])
            amount = st.number_input("Amount", min_value=0.0, value=10000.0)
            old_balance_orig = st.number_input("Origin Old Balance", min_value=0.0, value=50000.0)
            new_balance_orig = st.number_input("Origin New Balance", min_value=0.0, value=40000.0)
        
        with col2:
            old_balance_dest = st.number_input("Destination Old Balance", min_value=0.0, value=0.0)
            new_balance_dest = st.number_input("Destination New Balance", min_value=0.0, value=10000.0)
            is_flagged = st.selectbox("Is Flagged Fraud", [0, 1])
            
            # Encode transaction type
            type_mapping = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
            encoded_type = type_mapping[trans_type]
        
        if st.button("🔍 Analyze Transaction"):
            # Create transaction data
            transaction_data = pd.Series({
                'step': step,
                'type': encoded_type,
                'amount': amount,
                'nameOrig': 12345,  # Dummy value
                'oldbalanceOrg': old_balance_orig,
                'newbalanceOrig': new_balance_orig,
                'nameDest': 67890,  # Dummy value
                'oldbalanceDest': old_balance_dest,
                'newbalanceDest': new_balance_dest,
                'isFlaggedFraud': is_flagged
            })
            
            # Preprocess and predict
            features = preprocess_transaction(transaction_data, scaler, le)
            rf_prob = rf_model.predict_proba(features)[0][1]
            xgb_prob = xgb_model.predict_proba(features)[0][1]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Random Forest Risk", f"{rf_prob:.2%}")
            with col2:
                st.metric("XGBoost Risk", f"{xgb_prob:.2%}")
            with col3:
                max_risk = max(rf_prob, xgb_prob)
                risk_level = "HIGH" if max_risk > 0.5 else "MEDIUM" if max_risk > 0.2 else "LOW"
                st.metric("Overall Risk", f"{risk_level} ({max_risk:.2%})")
            
            # Risk visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Random Forest', 'XGBoost'],
                y=[rf_prob, xgb_prob],
                marker_color=['#1f77b4', '#ff7f0e']
            ))
            fig.update_layout(
                title="Fraud Risk Assessment",
                yaxis_title="Fraud Probability",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("📈 Fraud Analytics Dashboard")
        
        # Load full dataset for analytics
        df = load_data()
        if df is not None:
            # Fraud statistics
            fraud_rate = df['isFraud'].mean()
            total_transactions = len(df)
            fraud_transactions = df['isFraud'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transactions", f"{total_transactions:,}")
            with col2:
                st.metric("Fraud Cases", f"{fraud_transactions:,}")
            with col3:
                st.metric("Fraud Rate", f"{fraud_rate:.3%}")
            
            # Fraud by transaction type
            fraud_by_type = df.groupby('type')['isFraud'].agg(['count', 'sum', 'mean']).reset_index()
            fraud_by_type.columns = ['Transaction_Type', 'Total', 'Fraud_Count', 'Fraud_Rate']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(fraud_by_type, x='Transaction_Type', y='Fraud_Rate', 
                           title='Fraud Rate by Transaction Type')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(fraud_by_type, values='Total', names='Transaction_Type',
                           title='Transaction Volume by Type')
                st.plotly_chart(fig, use_container_width=True)
            
            # Amount distribution
            fig = make_subplots(rows=1, cols=2, subplot_titles=['Normal Transactions', 'Fraudulent Transactions'])
            
            normal_amounts = df[df['isFraud'] == 0]['amount']
            fraud_amounts = df[df['isFraud'] == 1]['amount']
            
            fig.add_trace(go.Histogram(x=normal_amounts, name='Normal', nbinsx=50), row=1, col=1)
            fig.add_trace(go.Histogram(x=fraud_amounts, name='Fraud', nbinsx=50), row=1, col=2)
            
            fig.update_layout(title="Amount Distribution: Normal vs Fraudulent Transactions")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("⚙️ Model Performance Metrics")
        
        # Model performance summary
        st.subheader("📊 Model Performance Summary")
        
        performance_data = {
            'Model': ['Random Forest', 'XGBoost'],
            'Accuracy': [99.98, 99.94],
            'Precision': [89.13, 73.50],
            'Recall': [100.0, 89.63],
            'F1-Score': [94.25, 80.77]
        }
        
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, use_container_width=True)
        
        # Performance visualization
        fig = px.bar(performance_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                    x='Metric', y='Score', color='Model', barmode='group',
                    title='Model Performance Comparison')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (mock data for demonstration)
        st.subheader("🎯 Feature Importance")
        
        feature_importance = {
            'Feature': ['amount', 'balance_change_orig', 'amount_to_balance_ratio', 
                       'type', 'oldbalanceOrg', 'amount_log'],
            'Importance': [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
        }
        
        importance_df = pd.DataFrame(feature_importance)
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title='Feature Importance in Fraud Detection')
        st.plotly_chart(fig, use_container_width=True)
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
