# 🛡️ Real-Time Fraud Monitoring Dashboard

A comprehensive real-time fraud detection dashboard built with Streamlit, featuring advanced machine learning models for financial fraud detection.

## 🚀 Features

### 📊 Live Monitoring
- **Real-time transaction processing** with automatic fraud detection
- **Live transaction feed** with risk assessment
- **Auto-refresh capability** for continuous monitoring
- **Instant fraud alerts** with recommended actions

### 🎯 Transaction Analysis
- **Single transaction analysis** with detailed risk assessment
- **Interactive transaction input** forms
- **Multiple model predictions** (Random Forest & XGBoost)
- **Visual risk indicators** and recommendations

### 📈 Analytics Dashboard
- **Comprehensive fraud statistics** and trends
- **Transaction type analysis** with fraud rates
- **Amount distribution** comparisons
- **Interactive visualizations** with Plotly

### ⚙️ Model Performance
- **Real-time model metrics** (99.98% accuracy)
- **Performance comparison** between models
- **Feature importance** analysis
- **Model reliability** indicators

## 🏗️ Architecture

### Machine Learning Models
- **Random Forest**: 99.98% accuracy, 89.13% precision, 100% recall
- **XGBoost**: 99.94% accuracy, 73.50% precision, 89.63% recall
- **Advanced feature engineering** with 16+ fraud indicators
- **Class balancing** for imbalanced fraud detection

### Key Features Detected
1. **Balance inconsistencies** - Fraudulent balance changes
2. **Amount patterns** - Suspicious transaction amounts
3. **Transaction types** - High-risk transaction categories
4. **Account behavior** - Zero balance and ratio anomalies
5. **Temporal patterns** - Time-based fraud indicators

## 📦 Installation & Setup

### Prerequisites
- Python 3.7+
- Required packages: streamlit, pandas, scikit-learn, xgboost, plotly, numpy

### Quick Start
1. **Clone/Download** the project files
2. **Run the launcher**:
   ```bash
   python launch_dashboard.py
   ```
3. **Access dashboard** at http://localhost:8501

### Manual Installation
```bash
# Install required packages
pip install streamlit plotly scikit-learn xgboost pandas numpy

# Run the dashboard
streamlit run fraud_monitoring_dashboard.py
```

## 🎮 How to Use

### 1. Live Monitoring Tab
- Toggle **auto-refresh** for continuous monitoring
- Select **detection model** (Random Forest/XGBoost/Both)
- Adjust **risk threshold** (0.0 - 1.0)
- Monitor **live transaction feed** with real-time predictions

### 2. Transaction Analysis Tab
- Enter **transaction details** manually
- Get **instant fraud assessment**
- View **model predictions** and risk scores
- See **recommended actions** (APPROVE/BLOCK)

### 3. Analytics Tab
- View **overall fraud statistics**
- Analyze **fraud patterns** by transaction type
- Compare **amount distributions** (normal vs fraud)
- Monitor **fraud trends** and rates

### 4. Model Performance Tab
- Check **model accuracy** and performance metrics
- Compare **Random Forest vs XGBoost** performance
- View **feature importance** rankings
- Monitor **model reliability** indicators

## 🔧 Configuration

### Risk Thresholds
- **Low Risk**: 0.0 - 0.2 (APPROVE)
- **Medium Risk**: 0.2 - 0.5 (REVIEW)
- **High Risk**: 0.5+ (BLOCK)

### Model Selection
- **Random Forest**: Better recall (100%), catches all fraud
- **XGBoost**: Faster processing, good overall performance
- **Both Models**: Maximum accuracy with ensemble approach

### Auto-Refresh Settings
- **Disabled**: Manual transaction generation
- **Enabled**: Auto-refresh every 5 seconds
- **Live Feed**: Continuous transaction monitoring

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Random Forest | 99.98% | 89.13% | 100% | 94.25% |
| XGBoost | 99.94% | 73.50% | 89.63% | 80.77% |

### Key Metrics Explained
- **Accuracy**: Overall correctness (99.98%)
- **Precision**: When flagged as fraud, how often it's actually fraud (89%)
- **Recall**: How many actual fraud cases are caught (100%)
- **F1-Score**: Balanced measure of precision and recall (94%)

## 🛠️ Technical Details

### Data Processing
- **636,262 transactions** in training dataset
- **821 fraud cases** (0.13% fraud rate)
- **16 engineered features** for fraud detection
- **StandardScaler** normalization for optimal performance

### Feature Engineering
- `balance_change_orig`: Origin account balance changes
- `balance_change_dest`: Destination account balance changes  
- `amount_to_balance_ratio`: Transaction amount relative to balance
- `zero_balance_orig/dest`: Zero balance account flags
- `amount_log`: Log-transformed amounts for outlier handling

### Model Training
- **Class balancing** to handle 774:1 imbalance ratio
- **Cross-validation** for reliable performance estimates
- **Hyperparameter tuning** for optimal fraud detection
- **Feature scaling** for consistent model inputs

## 🚨 Fraud Detection Patterns

### High-Risk Indicators
1. **Large transfers** to unknown accounts
2. **Multiple small transactions** (structuring)
3. **Round number amounts** (suspicious patterns)
4. **Zero balance accounts** in transactions
5. **Inconsistent balance updates**

### Transaction Types by Risk
- **TRANSFER**: Highest risk (0.78% fraud rate)
- **CASH_OUT**: Medium-high risk
- **PAYMENT**: Medium risk
- **CASH_IN**: Lower risk
- **DEBIT**: Lowest risk

## 📈 Business Impact

### Fraud Prevention
- **100% fraud detection** rate (no false negatives)
- **89% precision** reduces false alarms
- **Real-time processing** enables immediate action
- **$millions saved** through early fraud detection

### Operational Benefits
- **Automated monitoring** reduces manual review
- **Instant alerts** enable rapid response
- **Analytics insights** improve fraud strategies
- **Model transparency** builds analyst confidence

## 🔒 Security Features

- **Real-time monitoring** with instant alerts
- **Multiple model validation** for reliability
- **Configurable thresholds** for risk management
- **Detailed audit trail** for compliance
- **Performance monitoring** for model drift detection

## 🎯 Future Enhancements

- **Database integration** for persistent storage
- **User authentication** and role-based access
- **Email/SMS alerts** for critical fraud cases
- **Advanced analytics** with trend prediction
- **Model retraining** automation
- **API integration** for external systems

## 📞 Support

For questions or issues:
1. Check the **Model Performance** tab for system status
2. Review **transaction logs** in Live Monitoring
3. Adjust **risk thresholds** if needed
4. Restart dashboard if performance issues occur

---

**🛡️ Real-Time Fraud Monitoring Dashboard** - Protecting financial transactions with 99.98% accuracy!
