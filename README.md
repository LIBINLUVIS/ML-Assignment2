#  Metaverse Financial Fraud Detection

## Problem Statement
The objective of this project is to build a robust machine learning system capable of detecting **high-risk transactions** (fraud) in a Metaverse environment. By analyzing transaction patterns, login behaviors, and user demographics, the system aims to classify transactions as either legitimate or high-risk, thereby enhancing security and trust in virtual economies.

## Dataset Overview
The project utilizes a preprocessed dataset of metaverse transactions containing approximately **33,000 records**.
**Dataset Source:**  
Open Metaverse. *Metaverse Financial Transactions Dataset*. Kaggle.  
Available at: https://www.kaggle.com/datasets/faizaniftikharjanjua/metaverse-financial-transactions-dataset

### Class Distribution & Evaluation Strategy
The derived dataset exhibits a moderately balanced distribution, avoiding the extreme skew often found in traditional fraud datasets:
* **High-Risk Transactions (Class 1):** ~31.8%
* **Normal Transactions (Class 0):** ~68.2%

Given that the minority class representation is significant, the model can be trained without the need for synthetic resampling (e.g., SMOTE). Performance is evaluated using a comprehensive suite of metrics to ensure a holistic view of model effectiveness:
* **Accuracy & F1-Score:** For overall reliability.
* **Precision & Recall:** To measure the trade-off between false alarms and missed detections.
* **AUC & MCC:** To assess the model's discriminative power and correlation regardless of class size.

###  Model Benchmarking Observations
Preliminary testing across various algorithms provided the following insights:
* **Ensemble Models (XGBoost, Random Forest):** Provided superior results by effectively capturing non-linear patterns in metaverse transaction behavior.
* **k-Nearest Neighbors (kNN):** Demonstrated strong performance, though limited by scalability issues with larger transaction volumes.
* **Probabilistic Models (Naive Bayes):** Acted as a fast, baseline classifier but was outperformed by models capable of detecting complex feature interactions.

---
**Target Variable:**
*   `is_high_risk_transaction`: 0 (Legitimate), 1 (High Risk/Fraud)

**Key Features:**
*   **Transactional:** `amount`, `transaction_type`, `hour_of_day`
*   **Behavioral:** `login_frequency`, `session_duration`, `purchase_pattern`
*   **Demographic/Regional:** `age_group`, `location_region`
*   **Technical:** `ip_prefix` (anonymized), `anomaly` scores

## Solution Architecture
The solution involves a complete end-to-end machine learning pipeline:
1.  **Data Preprocessing**: Handling categorical variables (One-Hot Encoding) and numerical scaling (StandardScaler) using a Scikit-Learn Pipeline.
2.  **Model Training**: Training and evaluating six different classification algorithms:
    *   Logistic Regression
    *   Decision Tree
    *   K-Nearest Neighbors (KNN)
    *   Gaussian Naive Bayes
    *   Random Forest
    *   XGBoost
3.  **Evaluation**: continuous assessment using metrics like **ROC-AUC**, **F1-Score**, **Precision**, **Recall**, and **Matthews Correlation Coefficient (MCC)**.
4.  **Deployment**: A user-friendly **Streamlit** web application for real-time model interaction and testing.

## 🚀 Tech Stack
*   **Language**: Python 3.9+
*   **Machine Learning**: Scikit-Learn, XGBoost
*   **Data Manipulation**: Pandas, NumPy
*   **Visualization**: Matplotlib, Seaborn
*   **Web App**: Streamlit
*   **Model Serialization**: Joblib

## 📁 Project Structure
``` 
ML_Project/
├── app.py                      # Streamlit dashboard application
├── train_models.py             # Script to train and save models
├── eda_metaverse.py            # Exploratory Data Analysis script
├── requirements.txt            # Project dependencies
├── processed_metaverse_transactions.csv  # Dataset
├── models/                     # Directory for saved model artifacts (.pkl)
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore file
```

## Models Used

The following machine learning models were implemented and evaluated for fraud detection:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 94.98% | 99.15% | 91.51% | 92.84% | 92.17% | 88.48% |
| Decision Tree | 99.97% | 99.97% | 99.95% | 99.95% | 99.95% | 99.93% |
| KNN | 98.79% | 99.81% | 98.21% | 97.98% | 98.09% | 97.20% |
| Naive Bayes | 87.62% | 98.16% | 100.00% | 61.11% | 75.86% | 71.91% |
| Random Forest (Ensemble) | 99.96% | 100.00% | 99.95% | 99.91% | 99.93% | 99.90% |
| XGBoost (Ensemble) | 99.99% | 100.00% | 99.95% | 100.00% | 99.98% | 99.97% |

## Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Shows good baseline performance with balanced metrics. Suitable for interpretable fraud detection with decent precision-recall balance. |
| Decision Tree | Excellent performance with high interpretability. Nearly perfect scores across all metrics, making it highly effective for fraud detection. |
| KNN | Strong performance with high accuracy and AUC. Good for capturing local patterns in transaction data with consistent precision-recall balance. |
| Naive Bayes | High precision but lower recall indicates conservative fraud detection. Perfect precision means no false positives but misses some fraudulent transactions. |
| Random Forest (Ensemble) | Outstanding ensemble performance with near-perfect metrics. Combines multiple decision trees for robust and reliable fraud detection. |
| XGBoost (Ensemble) | Best overall performance with the highest scores across all metrics. Gradient boosting provides superior fraud detection capabilities with perfect recall. |

## 🚀 Deployment

This application is deployed on **Streamlit Cloud** and can be accessed at:
**[Live Demo](https://ml-assignment2-k5lxgyzxqklzfb9vlqqwb7.streamlit.app/)** *(URL will be available after deployment)*

### Local Development
To run the application locally:

1. Clone the repository:
```bash
git clone https://github.com/LIBINLUVIS/ML-Assignment2.git
cd ML-Assignment2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the models (if not already done):
```bash
python train_models.py
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Streamlit Cloud Deployment

The app is automatically deployed from the main branch of this repository. Any changes pushed to the main branch will trigger a new deployment.

## 📊 How to Use the App

1. **Select a Model**: Choose from the available trained models in the sidebar
2. **Upload Test Data**: Upload a CSV file containing transaction data with the required features
3. **Run Evaluation**: Click the "Run Evaluation" button to see model performance metrics
4. **View Results**: Analyze the confusion matrix, classification report, and performance metrics




