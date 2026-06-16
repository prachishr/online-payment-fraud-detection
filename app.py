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

def create_input_dataframe():

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

    # Feature Engineering

    df["balance_diff_orig"] = (
        df["oldbalanceorg"] - df["newbalanceorig"]
    )

    df["balance_diff_dest"] = (
        df["newbalancedest"] - df["oldbalancedest"]
    )

    df["amount_balance_ratio"] = (
        df["amount"] / (df["oldbalanceorg"] + 1)
    )

    df["orig_zero_after"] = (
        df["newbalanceorig"] == 0
    ).astype(int)

    df["dest_zero_before"] = (
        df["oldbalancedest"] == 0
    ).astype(int)

    df["is_cashout_or_transfer"] = (
        df["type"].isin(["CASHOUT", "TRANSFER"])
    ).astype(int)

    df["error_balance_orig"] = (
        df["oldbalanceorg"]
        - df["amount"]
        - df["newbalanceorig"]
    )

    df["error_balance_dest"] = (
        df["oldbalancedest"]
        + df["amount"]
        - df["newbalancedest"]
    )

    type_encoded = encoder.transform(df[['type']])

    type_df = pd.DataFrame(
        type_encoded,
        columns=encoder.get_feature_names_out(['type']),
        index=df.index
    )

    df = pd.concat([df, type_df], axis=1)
    df = df.drop('type', axis=1)

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

        probability = model.predict_proba(input_df)[0][1]

        
        THRESHOLD = 0.9

        if probability >= THRESHOLD:
            prediction = 1
        else:
            prediction = 0

        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if prediction >= 1:
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
        
        if probability > THRESHOLD:
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
