import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Online Payment Fraud Detection",
    page_icon="🔒",
    layout="wide"
)

# Title
st.title("🔒 Online Payment Fraud Detection System")
st.markdown("""
This application uses a **XGBoost   Classifier** to predict whether an online payment transaction is fraudulent or legitimate.
Enter the transaction details below to get a prediction.
""")

# Load the trained models
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open("xgb_fraud_model.pkl", "rb"))
        encoder = pickle.load(open("encoder.pkl", "rb"))
        features = pickle.load(open("features.pkl", "rb"))
        return model, encoder, features
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please ensure 'xgb_fraud_model.pkl', 'encoder.pkl', and 'features.pkl' are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

model, encoder, feature_columns = load_models()

# Define IQR bounds (calculated from your training data)
# These should match the clipping values used during training
def get_iqr_bounds():
    """Return IQR bounds for outlier clipping - adjust these values based on your training data"""
    # These are approximate bounds - you should extract exact values from your training data
    bounds = {
        'amount': (0, 2000000),  # Adjust based on your data
        'oldbalanceorg': (0, 50000000),
        'newbalanceorig': (0, 50000000),
        'oldbalancedest': (0, 50000000),
        'newbalancedest': (0, 50000000),
    }
    return bounds

# Sidebar for input
st.sidebar.header("📊 Transaction Details")

# Transaction type selection
transaction_types = {
    "CASHIN": "Cash In",
    "CASHOUT": "Cash Out", 
    "DEBIT": "Debit",
    "PAYMENT": "Payment",
    "TRANSFER": "Transfer"
}

# Input fields
transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    options=list(transaction_types.keys()),
    format_func=lambda x: transaction_types[x]
)

amount = st.sidebar.number_input(
    "Transaction Amount",
    
    value=50000.0,
    step=10000.0
)

oldbalanceOrg = st.sidebar.number_input(
    "Original Balance (Origin Account)",
    
    value=100000.0,
    step=50000.0
)

newbalanceOrig = st.sidebar.number_input(
    "New Balance (Origin Account)",
    
    value=50000.0,
    step=50000.0
)

oldbalanceDest = st.sidebar.number_input(
    "Original Balance (Destination Account)",
    
    value=50000.0,
    step=50000.0
)

newbalanceDest = st.sidebar.number_input(
    "New Balance (Destination Account)",
    
    value=100000.0,
    step=50000.0
)

step = st.sidebar.number_input(
    "Time Step (hour of transaction)",
    min_value=1,
    max_value=744,
    value=200,
    step=10
)

isFlaggedFraud = st.sidebar.selectbox(
    "Flagged as Fraud by System",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

def apply_outlier_clipping(df, bounds):
    """Apply same outlier clipping as during training"""
    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower, upper)
    return df

# Create input dataframe
def create_input_dataframe():
    """Create a DataFrame from user inputs with proper preprocessing"""
    input_data = {
        'step': step,
        'amount': amount,
        'oldbalanceorg': oldbalanceOrg,
        'newbalanceorig': newbalanceOrig,
        'oldbalancedest': oldbalanceDest,
        'newbalancedest': newbalanceDest,
        'isflaggedfraud': isFlaggedFraud,
        'type': transaction_type
    }
    
    df = pd.DataFrame([input_data])
    
    # Apply outlier clipping (IMPORTANT - matches training preprocessing)
    bounds = get_iqr_bounds()
    df = apply_outlier_clipping(df, bounds)
    
    # One-hot encode the type column
    type_encoded = encoder.transform(df[['type']])
    type_df = pd.DataFrame(
        type_encoded,
        columns=encoder.get_feature_names_out(['type']),
        index=df.index
    )
    
    # Concatenate and drop original type column
    df = pd.concat([df, type_df], axis=1)
    df = df.drop('type', axis=1)
    
    # Reindex to match training features
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    return df

# Prediction button
predict_button = st.sidebar.button("🔍 Predict Fraud", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ **Note**: Based on transaction patterns, CASH_OUT and TRANSFER transactions have higher fraud risk."
)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Transaction Summary")
    st.markdown(f"""
    - **Transaction Type:** {transaction_types[transaction_type]}
    - **Amount:** {amount:,.2f}
    - **Time Step:** {step}
    - **Flagged:** {'Yes' if isFlaggedFraud == 1 else 'No'}
    """)

with col2:
    st.subheader("💰 Balance Information")
    st.markdown(f"""
    - **Origin Account:** {oldbalanceOrg:,.2f} → {newbalanceOrig:,.2f}
    - **Destination Account:** {oldbalanceDest:,.2f} → {newbalanceDest:,.2f}
    """)

# Make prediction
if predict_button:
    try:
        input_df = create_input_dataframe()
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if prediction == 1:
                st.error("⚠️ FRAUD ALERT! This transaction appears FRAUDULENT")
            else:
                st.success("✅ This transaction appears LEGITIMATE")
        
        with col_res2:
            fraud_prob = probability * 100
            legit_prob = (1 - probability) * 100
            st.metric("Fraud Probability", f"{fraud_prob:.2f}%")
        
        # Risk factors
        st.markdown("---")
        st.subheader("📊 Risk Assessment")
        
        risk_factors = []
        
        if transaction_type in ["CASHOUT", "TRANSFER"]:
            risk_factors.append("⚠️ High-risk transaction type (CASH_OUT/TRANSFER)")
        
        # Check balance consistency
        if abs(oldbalanceOrg - newbalanceOrig - amount) > 1:
            risk_factors.append("❌ Balance mismatch in origin account")
        
        if probability > 0.5:
            risk_factors.append(f"🔴 High fraud probability ({fraud_prob:.1f}%)")
        
        if risk_factors:
            for factor in risk_factors:
                st.write(factor)
        else:
            st.write("✅ No significant risk factors detected")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendation")
        
        if prediction == 1:
            st.warning("""
            **Immediate Action Required:**
            1. Flag transaction for review
            2. Verify with account holders
            3. Hold transaction pending investigation
            """)
        else:
            st.info("Transaction can be processed normally. Continue monitoring for unusual patterns.")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Fraud Detection System | XGBoost Classifier</small></center>",
    unsafe_allow_html=True
)
