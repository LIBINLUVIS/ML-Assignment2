
import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# Configuration
MODEL_DIR = 'models'
TARGET_COLUMN = 'is_high_risk_transaction'  # Assumed target column name

st.set_page_config(page_title="Fraud Detection Model Evaluator", layout="wide")

st.title("Financial Fraud Detection - Model Evaluator")

# 1. Sidebar - Model Selection
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')] if os.path.exists(MODEL_DIR) else []
model_map = {name.replace('_', ' ').replace('.pkl', '').title(): name for name in model_files}

st.sidebar.header("Configuration")
selected_model_name = st.sidebar.selectbox("Select Model", list(model_map.keys()))

# 2. Upload Data
st.sidebar.subheader("Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain features + target)", type=["csv"])

def load_model(model_filename):
    path = os.path.join(MODEL_DIR, model_filename)
    return joblib.load(path)

# Main Execution
if uploaded_file and selected_model_name:
    if st.button("Run Evaluation"):
        try:
            # Load Data
            df = pd.read_csv(uploaded_file)
            
            # check if target exists
            if TARGET_COLUMN not in df.columns:
                st.error(f"Dataset must contain the target column '{TARGET_COLUMN}' for evaluation.")
            else:
                X_test = df.drop(columns=[TARGET_COLUMN])
                y_test = df[TARGET_COLUMN]
                
                # Load Model
                model_filename = model_map[selected_model_name]
                model = load_model(model_filename)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_proba)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                mcc = matthews_corrcoef(y_test, y_pred)
                
                # Display Metrics
                st.subheader(f"Results for {selected_model_name}")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("ROC-AUC", f"{roc_auc:.4f}")
                col3.metric("Precision", f"{prec:.4f}")
                col4.metric("Recall", f"{rec:.4f}")
                col5.metric("F1-Score", f"{f1:.4f}")
                col6.metric("MCC", f"{mcc:.4f}")
                
                # Visualization Layout
                col_graph1, col_graph2 = st.columns([1, 2])
                
                with col_graph1:
                    st.markdown("#### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    st.pyplot(fig)
                    
                with col_graph2:
                    st.markdown("#### Classification Report")
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df.style.format("{:.4f}"))
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")
elif not uploaded_file:
    st.info("Please upload a CSV file to proceed.")
elif not os.path.exists(MODEL_DIR) or not model_files:
    st.warning("No trained models found. Please run 'train_models.py' first.")
