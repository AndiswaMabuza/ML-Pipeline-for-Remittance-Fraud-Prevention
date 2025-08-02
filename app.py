import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTENC 


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="ML Pipeline For Remittance Fraud Prevention",
    page_icon="🕵️‍♂️",
    layout="wide", # Use wide layout for better display
    initial_sidebar_state="expanded"
)

# --- Dark Mode Hint ---
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    /* You can set dark mode via .streamlit/config.toml:
       [theme]
       base="dark"
       primaryColor="#F63366"
       backgroundColor="#0E1117"
       secondaryBackgroundColor="#262730"
       textColor="#FAFAFA"
       font="sans serif"
    */
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Pre-trained Assets ---
@st.cache_resource # Use st.cache_resource for models and large objects
def load_assets():
    try:
        model = joblib.load('fraud_detection_model.pkl')
        preprocessor = joblib.load('data_preprocessor.pkl')
        threshold = joblib.load('classification_threshold.pkl')
        all_feature_names = joblib.load('all_feature_names.pkl')
        numerical_cols = joblib.load('numerical_cols.pkl')
        categorical_cols = joblib.load('categorical_cols.pkl')
        return model, preprocessor, threshold, all_feature_names, numerical_cols, categorical_cols
    except FileNotFoundError:
        st.error("Model assets not found! Please ensure 'fraud_detection_model.pkl', 'data_preprocessor.pkl', 'classification_threshold.pkl', 'all_feature_names.pkl', 'numerical_cols.pkl', and 'categorical_cols.pkl' are in the same directory.")
        st.stop()

model, preprocessor, classification_threshold, all_feature_names, numerical_cols, categorical_cols = load_assets()


# --- Helper Function for Prediction ---
def get_fraud_prediction(input_data_dict, model, preprocessor, numerical_cols, categorical_cols, all_feature_names, threshold):
    # Convert input dict to pandas Series, then to DataFrame row
    transaction_series = pd.Series(input_data_dict)
    transaction_df = pd.DataFrame([transaction_series])

    # Ensure categorical columns are strings for CatBoost
    for col in categorical_cols:
        if col in transaction_df.columns:
            transaction_df[col] = transaction_df[col].astype(str)

    # Separate numerical and categorical parts
    num_data = transaction_df[numerical_cols]
    cat_data = transaction_df[categorical_cols]

    # Apply numerical preprocessing
    num_processed = preprocessor.transform(num_data)
    num_processed_df = pd.DataFrame(num_processed, columns=numerical_cols, index=transaction_df.index)

    # Combine processed numerical features with original categorical features
    # Ensure correct column order as expected by the trained model
    combined_df = pd.concat([num_processed_df.reset_index(drop=True), cat_data.reset_index(drop=True)], axis=1)
    combined_df = combined_df[all_feature_names] # Reorder columns to match training


    # Get fraud probability from the model
    fraud_probability = model.predict_proba(combined_df)[:, 1][0]

    # Determine prediction based on threshold
    prediction = 1 if fraud_probability >= threshold else 0

    return fraud_probability, prediction


# --- Main Application Layout ---
st.title("🛡️ ML Pipeline For Remittance Fraud Prevention")
st.markdown("---")

st.header("Project Overview & Journey to Exceptional Performance")
st.write(
    """
    This application demonstrates a highly optimized machine learning model for detecting fraudulent financial transactions.
    The journey involved tackling a common challenge in fraud detection: **extremely imbalanced data** (very few fraud cases).
    Our goal was dual-pronged:
    1.  **Maximize Recall (Catch all fraud):** Missing a single fraudulent transaction can be disastrous for clients and institutions.
    2.  **Maximize Precision (Minimize False Positives):** Too many false alarms burden operational teams and frustrate legitimate clients.
    """
)

st.subheader("The Journey:")
st.markdown(
    """
    1.  **Initial Attempt (Logistic Regression with SMOTE):**
        * I started with a Logistic Regression model on SMOTE-resampled data.
        * **Results:** While we achieved **1.00 Recall**, my **Precision was extremely low at just 0.18**. This meant catching all fraud, but at the cost of flagging a massive number of legitimate transactions as fraudulent, an unsustainable operational burden.

    2.  **Optimization with Threshold Tuning:**
        * Recognizing the model's excellent **ROC AUC (0.9993)**, I knew the model was good at *ranking* transactions.
        * I optimized the classification threshold to maximize the F1-score (a balance of precision and recall).
        * **Results:** Precision significantly improved to **0.45**, with Recall at **0.90**. This was a much better balance, reducing false positives while still catching most fraud.

    3.  **Breakthrough with CatBoost:**
        * To achieve truly exceptional performance, I transitioned to **CatBoost**, a state-of-the-art gradient boosting algorithm known for its robust performance on tabular data and its native handling of categorical features (common in transaction data).
        * The preprocessing pipeline was adapted to leverage CatBoost's capabilities (using SMOTENC and passing categorical features directly).
    """
)

st.subheader("Current Model Performance (CatBoost):")
st.markdown("The CatBoost model, with an optimized threshold of **`{:.4f}`**, achieves near-perfect performance:".format(classification_threshold))

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Fraud Recall (Class 1)", value="1.00")
    st.caption("All fraudulent transactions are detected.")
with col2:
    st.metric(label="Fraud Precision (Class 1)", value="0.77")
    st.caption("77% of flagged transactions are actually fraud.")
with col3:
    st.metric(label="Fraud F1-Score (Class 1)", value="0.87")
    st.caption("Excellent balance between precision and recall.")
with col4:
    st.metric(label="ROC AUC Score", value="0.9998")
    st.caption("Near-perfect ability to distinguish fraud from non-fraud.")

st.markdown(
    """
    This level of performance ensures that clients and institutions are protected from financial losses due to fraud (100% Recall),
    while simultaneously keeping the operational burden of reviewing false positives to a minimum (high Precision).
    """
)

st.markdown("---")

st.header("Interactive Fraud Risk Assessment")
st.write("Enter transaction details below to get a real-time fraud risk assessment.")

# --- Input Fields for Transaction Details ---
with st.form("transaction_form"):
    st.subheader("Transaction Details")

    # Arrange numerical inputs in columns
    num_cols_display = st.columns(len(numerical_cols))
    input_numerical_data = {}
    for i, col in enumerate(numerical_cols):
        with num_cols_display[i]:
            # Use float_input for numerical columns, with a default for easy testing
            default_value = 100.0 if "amount" in col else 1.0
            input_numerical_data[col] = st.number_input(f"Enter {col.replace('_', ' ').title()}", value=default_value, step=0.1)

    st.subheader("Categorical Details")
    # Arrange categorical inputs in columns
    cat_cols_display = st.columns(len(categorical_cols))
    input_categorical_data = {}
    for i, col in enumerate(categorical_cols):
        with cat_cols_display[i]:
            # Use selectbox for categorical columns with example options
            # Replace with actual unique values from your data if known/desired for dropdown
            if col == 'channel':
                options = ['Online', 'POS', 'ATM', 'Mobile']
            elif col == 'device_type':
                options = ['Mobile', 'Desktop', 'Tablet']
            elif col == 'location':
                options = ['Local', 'Domestic', 'International']
            elif col == 'payment_method':
                options = ['Credit Card', 'Debit Card', 'Bank Transfer', 'E-wallet']
            elif col == 'txn_type':
                options = ['Purchase', 'Withdrawal', 'Transfer', 'Refund']
            else:
                options = ['Category A', 'Category B', 'Category C'] # Placeholder if unknown

            input_categorical_data[col] = st.selectbox(f"Select {col.replace('_', ' ').title()}", options)

    submitted = st.form_submit_button("Assess Fraud Risk")

    if submitted:
        input_data = {**input_numerical_data, **input_categorical_data}

        # Get prediction
        fraud_prob, prediction = get_fraud_prediction(
            input_data, model, preprocessor, numerical_cols, categorical_cols, all_feature_names, classification_threshold
        )

        st.markdown("---")
        st.subheader("Assessment Results:")
        if prediction == 1:
            st.error(f"⚠️ **FLAGGED AS POTENTIAL FRAUD!** (Probability: {fraud_prob:.4f})")
            st.write("This transaction exceeds the fraud threshold and requires immediate review.")
        else:
            st.success(f"✅ **TRANSACTION APPEARS LEGITIMATE.** (Probability: {fraud_prob:.4f})")
            st.write("This transaction falls below the fraud threshold and can proceed.")

        st.write("---")
        st.info("Note: The model's decision threshold is `{:.4f}`. Transactions with a probability above this are flagged as fraud.".format(classification_threshold))
