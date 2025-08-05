
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard Title
st.title("🔒 Real-time Fraud Detection Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.title("Dashboard Controls")

# Load sample data (you would load your model and real-time data here)
@st.cache_data
def load_data():
    # In production, this would load from your database
    return pd.read_csv('synthetic_financial_data.csv')

df = load_data()

# Add fraud probability (simulated)
np.random.seed(42)
df['fraud_probability'] = np.random.beta(2, 5, len(df))

# Sidebar filters
st.sidebar.subheader("Filters")
risk_threshold = st.sidebar.slider("Fraud Risk Threshold", 0.0, 1.0, 0.5, 0.01)
selected_card_types = st.sidebar.multiselect("Card Types", df['card_type'].unique(), default=df['card_type'].unique())
selected_categories = st.sidebar.multiselect("Purchase Categories", df['purchase_category'].unique(), default=df['purchase_category'].unique())

# Filter data
filtered_df = df[
    (df['card_type'].isin(selected_card_types)) & 
    (df['purchase_category'].isin(selected_categories))
]

# High-risk transactions
high_risk_df = filtered_df[filtered_df['fraud_probability'] >= risk_threshold]

# Main dashboard metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Transactions", f"{len(filtered_df):,}")

with col2:
    st.metric("High Risk Transactions", f"{len(high_risk_df):,}", delta=f"{len(high_risk_df)/len(filtered_df)*100:.1f}%")

with col3:
    avg_amount = filtered_df['amount'].mean()
    st.metric("Average Amount", f"${avg_amount:,.2f}")

with col4:
    confirmed_fraud = filtered_df['is_fraudulent'].sum()
    st.metric("Confirmed Fraud", f"{confirmed_fraud:,}")

st.markdown("---")

# Charts row 1
col1, col2 = st.columns(2)

with col1:
    # Fraud probability distribution
    fig1 = px.histogram(filtered_df, x='fraud_probability', nbins=30, 
                       title="Fraud Probability Distribution")
    fig1.add_vline(x=risk_threshold, line_dash="dash", line_color="red", 
                   annotation_text="Risk Threshold")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Risk by card type
    risk_by_card = filtered_df.groupby('card_type')['fraud_probability'].mean().reset_index()
    fig2 = px.bar(risk_by_card, x='card_type', y='fraud_probability',
                  title="Average Risk by Card Type")
    st.plotly_chart(fig2, use_container_width=True)

# Charts row 2
col1, col2 = st.columns(2)

with col1:
    # Transaction amount vs fraud probability
    fig3 = px.scatter(filtered_df.sample(1000), x='amount', y='fraud_probability', 
                     color='is_fraudulent', title="Amount vs Fraud Probability")
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    # Risk by purchase category
    risk_by_category = filtered_df.groupby('purchase_category')['fraud_probability'].mean().reset_index()
    fig4 = px.pie(risk_by_category, values='fraud_probability', names='purchase_category',
                  title="Risk Distribution by Category")
    st.plotly_chart(fig4, use_container_width=True)

# High-risk transactions table
st.subheader("High-Risk Transactions Requiring Action")
if len(high_risk_df) > 0:
    display_cols = ['transaction_id', 'customer_id', 'amount', 'card_type', 
                   'purchase_category', 'fraud_probability', 'is_fraudulent']

    # Add action buttons
    high_risk_display = high_risk_df[display_cols].head(20)
    high_risk_display['fraud_probability'] = high_risk_display['fraud_probability'].round(3)

    st.dataframe(high_risk_display, use_container_width=True)

    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Investigate Selected"):
            st.success("Investigation initiated for selected transactions")
    with col2:
        if st.button("Block Transactions"):
            st.warning("Selected transactions have been blocked")
    with col3:
        if st.button("Mark as Reviewed"):
            st.info("Transactions marked as reviewed")
else:
    st.info("No high-risk transactions found with current threshold")

# Real-time monitoring section
st.subheader("Real-time Monitoring")
col1, col2 = st.columns(2)

with col1:
    # Simulated real-time data
    time_data = pd.date_range(start='2023-01-01', periods=24, freq='H')
    fraud_rates = np.random.beta(2, 10, 24)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=time_data, y=fraud_rates, mode='lines+markers', name='Fraud Rate'))
    fig5.update_layout(title="Fraud Rate Over Time", xaxis_title="Time", yaxis_title="Fraud Rate")
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    # Alert system
    st.subheader("Alert System")

    current_risk = np.random.beta(2, 5)
    if current_risk > 0.7:
        st.error(f"HIGH RISK ALERT: Current fraud rate {current_risk:.2%}")
    elif current_risk > 0.4:
        st.warning(f"MEDIUM RISK: Current fraud rate {current_risk:.2%}")
    else:
        st.success(f"LOW RISK: Current fraud rate {current_risk:.2%}")

    # Recent alerts
    st.subheader("Recent Alerts")
    alerts = [
        "High-value transaction detected - Customer ID 1234",
        "Unusual spending pattern - Customer ID 5678", 
        "Multiple failed attempts - Customer ID 9012"
    ]

    for alert in alerts:
        st.warning(alert)

# Footer
st.markdown("---")
st.markdown("Dashboard last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
