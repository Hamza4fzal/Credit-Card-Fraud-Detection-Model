import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# --- Load Model and Data ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        scaler_time = joblib.load('scaler_time.pkl')
        scaler_amount = joblib.load('scaler_amount.pkl')
        return model, scaler_time, scaler_amount
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv('creditcard.csv')
        fraud_data = df[df['Class'] == 1].sample(50)
        normal_data = df[df['Class'] == 0].sample(50)
        return fraud_data, normal_data
    except FileNotFoundError:
        return None, None

@st.cache_data
def get_class_distribution():
    try:
        # Load only the Class column to save memory
        df = pd.read_csv('creditcard.csv', usecols=['Class'])
        class_counts = df['Class'].value_counts().rename(index={0: 'Valid (0)', 1: 'Fraud (1)'})
        return class_counts
    except FileNotFoundError:
        return None

model, scaler_time, scaler_amount = load_model()
fraud_data, normal_data = load_sample_data()

# --- Functions ---
def log_transaction(transaction_data, prediction, probability):
    """Stores the transaction log in a CSV file."""
    log_file = 'transaction_logs.csv'
    
    # Create dictionary for the log
    log_entry = {
        'Timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Amount': transaction_data['Amount'],
        'Prediction': 'Fraud' if prediction == 1 else 'Valid',
        'Fraud_Probability': probability
    }
    
    # Add V1-V28 features
    for i in range(1, 29):
        log_entry[f'V{i}'] = transaction_data[f'V{i}']
        
    df_log = pd.DataFrame([log_entry])
    
    if not os.path.exists(log_file):
        df_log.to_csv(log_file, index=False)
    else:
        df_log.to_csv(log_file, mode='a', header=False, index=False)

# --- UI Layout ---
st.title(" Credit Card Fraud Detection System")
st.markdown("""
This application uses a Machine Learning model to detect fraudulent credit card transactions. 
You can simulate a dummy transaction to see the model in action.
""")

if model is None:
    st.error("Model not found! Please run `preprocessing.py` and `model_training.py` first.")
    st.stop()

# --- Sidebar for Navigation/Logs ---
with st.sidebar:
    st.header("Transaction Logs")
    st.write("Recent predictions are stored here.")
    if os.path.exists('transaction_logs.csv'):
        logs = pd.read_csv('transaction_logs.csv')
        st.dataframe(logs.tail(10)[['Timestamp', 'Amount', 'Prediction']])
        
        # Flagging Anomalies
        fraud_count = len(logs[logs['Prediction'] == 'Fraud'])
        if fraud_count > 0:
            st.warning(f"⚠️ {fraud_count} anomalous (fraudulent) transactions detected in logs!")
        
        if st.button("Clear Logs"):
            os.remove('transaction_logs.csv')
            st.rerun()
    else:
        st.info("No transactions logged yet.")

# --- Main Interaction Area ---
tab1, tab2 = st.tabs([" Predictor", " Data Visualization"])

with tab1:
    st.subheader("Perform a Dummy Transaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Option 1: Generate from existing patterns")
        st.write("Auto-fill features with data known to be valid or fraudulent for testing.")
        
        if st.button("Generate Dummy VALID Transaction"):
            if normal_data is not None:
                sample = normal_data.sample(1).iloc[0]
                st.session_state['dummy_data'] = sample
                st.success("Loaded normal transaction data!")
    
        if st.button("Generate Dummy FRAUDULENT Transaction"):
            if fraud_data is not None:
                sample = fraud_data.sample(1).iloc[0]
                st.session_state['dummy_data'] = sample
                st.error("Loaded fraudulent transaction data!")
    
    with col2:
        st.write("### Option 2: Manual Input")
        st.write("Tweak the transaction amount and see the result.")
        
        if 'dummy_data' not in st.session_state:
            # Default empty values
            default_time = 0.0
            default_amount = 0.0
            default_v = {f'V{i}': 0.0 for i in range(1, 29)}
        else:
            default_time = float(st.session_state['dummy_data']['Time'])
            default_amount = float(st.session_state['dummy_data']['Amount'])
            default_v = {f'V{i}': float(st.session_state['dummy_data'][f'V{i}']) for i in range(1, 29)}
    
        input_amount = st.number_input("Transaction Amount ($)", value=default_amount, format="%.2f")
    
    # Features input for prediction
    features = {'Time': default_time}
    features.update(default_v)
    features['Amount'] = input_amount
    
    st.divider()
    
    # Prediction Section
    if st.button("Predict Transaction Status", type="primary", use_container_width=True):
        with st.spinner("Analyzing transaction..."):
            # Create DataFrame for prediction
            input_df = pd.DataFrame([features])
            
            # Scale Time and Amount
            input_df['Time'] = scaler_time.transform(input_df['Time'].values.reshape(-1, 1))
            input_df['Amount'] = scaler_amount.transform(input_df['Amount'].values.reshape(-1, 1))
            
            # Make Prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1] # Probability of being fraud
            
            # Log it
            log_transaction(features, prediction, probability)
            
            # Display Result
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f" FRAUD DETECTED! ")
                st.write(f"The model is {probability*100:.2f}% confident this is a fraudulent transaction.")
            else:
                st.success(f" VALID TRANSACTION")
                st.write(f"The model is {(1-probability)*100:.2f}% confident this is a valid transaction.")
            
            st.info("Transaction has been logged.")

with tab2:
    st.subheader("Dataset Class Distribution")
    st.write("""
    This bar chart illustrates the extreme class imbalance present in the credit card fraud dataset. 
    Because fraudulent transactions are so rare, techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) 
    are required during preprocessing to train the Machine Learning model effectively.
    """)
    
    class_counts = get_class_distribution()
    if class_counts is not None:
        # Streamlit's built-in bar chart
        st.bar_chart(class_counts, color="#ff4b4b")
    else:
        st.error("Dataset not found. Please ensure 'creditcard.csv' is present.")
