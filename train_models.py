
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Configuration
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Define feature groups
NUMERICAL_FEATURES = ['hour_of_day', 'amount', 'ip_prefix', 'login_frequency', 'session_duration']
CATEGORICAL_FEATURES = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group', 'anomaly']
TARGET = 'is_high_risk_transaction'

def load_data():
    """
    Load the specific metaverse transactions dataset.
    """
    data_path = 'processed_metaverse_transactions.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Ensure target is separated
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    return X, y

def get_preprocessor():
    """
    Creates a preprocessor pipeline for numeric and categorical features.
    Drops unused columns (like timestamp, addresses) automatically via remainder='drop'.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop' 
    )
    return preprocessor

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model and return metrics.
    """
    # Pipeline handles preprocessing, so we pass raw X_test
    y_pred = model.predict(X_test)
    
    # Handle probability calculation
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except:
            y_proba = y_pred # Fallback if issues
    else:
        y_proba = y_pred

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    return metrics, y_pred

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def main():
    # 1. Load Data
    print("Loading data...")
    X, y = load_data()
    
    # 2. Split Data (Stratified 80/20)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # Save test data for the Streamlit app demo
    test_data = X_test.copy()
    test_data[TARGET] = y_test
    test_data.to_csv('test_data_sample.csv', index=False)
    print("Saved 'test_data_sample.csv' for Streamlit app usage.")

    # 3. Define Models (wrapped in pipeline)
    preprocessor = get_preprocessor()
    
    base_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Gaussian Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, use_label_encoder=False)
    }
    
    results = []
    
    # 4. Train and Evaluate
    print("Training and evaluating models...")
    for name, model_instance in base_models.items():
        print(f"Processing {name}...")
        
        # Create Pipeline
        # Note: GaussianNB with sparse input from OneHot might fail? 
        # GaussianNB requires dense. OneHotEncoder(sparse_output=True) is default.
        # We should check if we need sparse_output=False for NB.
        # Actually simplest to set sparse_output=False globally or use ToDense transformer.
        # For simplicity, we'll try standard. Only XGB/RF handle sparse well. NB usually needs dense.
        # Let's adjust preprocessor for NB specifically or make it dense for all (dataset is 33k rows, manageable).
        
        if name == 'Gaussian Naive Bayes':
             # Custom preprocessor for NB to ensure dense
             cat_trans_dense = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
             ])
             prep_dense = ColumnTransformer(
                transformers=[
                    ('num', Pipeline(steps=[('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERICAL_FEATURES),
                    ('cat', cat_trans_dense, CATEGORICAL_FEATURES)
                ], remainder='drop'
             )
             model = Pipeline(steps=[('preprocessor', prep_dense), ('classifier', model_instance)])
        else:
             # Standard handling
             model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model_instance)])

        # Train
        model.fit(X_train, y_train)
        
        # Save Model
        joblib.dump(model, os.path.join(MODEL_DIR, f'{name.replace(" ", "_").lower()}.pkl'))
        
        # Evaluate
        metrics, y_pred = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
        
        # Print Classification Report
        print(f"\n--- {name} Classification Report ---")
        print(classification_report(y_test, y_pred))

    # 5. Create Comparison Table
    results_df = pd.DataFrame(results).set_index('Model')
    print("\n=== Model Comparison Table ===")
    print(results_df)
    
    # Save results
    results_df.to_csv('model_comparison_results.csv')
    print("\nResults saved to 'model_comparison_results.csv'.")
    print(f"Models saved in '{MODEL_DIR}/'.")

if __name__ == "__main__":
    main()
