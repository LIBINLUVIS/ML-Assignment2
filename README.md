
#  Metaverse Financial Fraud Detection

## Problem Statement
The objective of this project is to build a robust machine learning system capable of detecting **high-risk transactions** (fraud) in a Metaverse environment. By analyzing transaction patterns, login behaviors, and user demographics, the system aims to classify transactions as either legitimate or high-risk, thereby enhancing security and trust in virtual economies.

## Dataset Overview
The project utilizes a preprocessed dataset of metaverse transactions containing approximately **33,000 records**.
**Dataset Source:**  
Open Metaverse. *Metaverse Financial Transactions Dataset*. Kaggle.  
Available at: https://www.kaggle.com/datasets/faizaniftikharjanjua/metaverse-financial-transactions-dataset

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


