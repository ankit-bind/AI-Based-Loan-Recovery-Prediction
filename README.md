
# AI-Based Loan Recovery Probability Prediction

## Overview

This project is an AI-powered Loan Recovery Probability Prediction system developed using Machine Learning techniques. The objective is to predict whether a loan is likely to be recovered based on borrower financial behavior, historical bureau records, repayment patterns, and social risk indicators.

The project focuses on advanced feature engineering, behavioral risk analysis, classification modeling, and explainable AI techniques to support optimized debt recovery strategies.

---

# Problem Statement

Financial institutions face significant challenges in identifying risky borrowers and optimizing loan recovery processes. Incorrect recovery decisions can increase financial losses and operational costs.

This project aims to build a predictive system that can:

- Predict loan recovery probability
- Analyze borrower repayment behavior
- Support risk-based recovery decisions
- Improve recovery strategy efficiency

---

# Dataset Used

The project uses the Home Credit Risk dataset containing:

- Borrower financial information
- Loan application details
- Bureau credit history
- Repayment behavior
- Social risk indicators

---

# Project Workflow

## 1. Data Collection
- Borrower application dataset
- Historical bureau dataset

## 2. Data Cleaning
- Missing value handling
- Anomaly detection
- Outlier treatment
- Duplicate checks

## 3. Feature Engineering
- Financial ratio features
- Bureau behavioral features
- Employment features
- Social risk features
- External credit score aggregation

## 4. Encoding & Transformation
- Binary Encoding
- One-Hot Encoding
- Frequency Encoding

## 5. Data Validation
- Correlation analysis
- Sanity checks
- Data quality verification

## 6. Model Building
- Classification modeling
- Recovery probability prediction
- ROC-AUC optimization

## 7. Explainable AI
- Feature importance analysis
- SHAP explainability

## 8. Deployment
- Streamlit web application
- Streamlit Cloud deployment

---

# Key Features Engineered

### Financial Features
- CREDIT_INCOME_RATIO
- ANNUITY_INCOME_RATIO
- GOODS_CREDIT_RATIO

### Bureau Behavioral Features
- ACTIVE_LOAN_RATIO
- CLOSED_LOAN_RATIO
- CREDIT_UTILIZATION_RATIO
- OVERDUE_PER_LOAN

### Behavioral Features
- EMPLOYMENT_AGE_RATIO
- AGE_YEARS
- EMPLOYMENT_YEARS

### Social Risk Features
- TOTAL_SOCIAL_DEFAULTS
- TOTAL_SOCIAL_OBS

### External Risk Features
- EXT_SOURCE_MEAN

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Seaborn

---

# Model Evaluation Metrics

- ROC-AUC Score
- Recall Score
- Precision
- F1-Score
- Confusion Matrix

---

# Expected Outcome

The final system predicts the probability of successful loan recovery and helps financial institutions make better recovery decisions using AI-driven insights.

---

# Future Improvements

- Hyperparameter tuning
- Ensemble learning
- Deep Learning models
- Real-time API integration
- Advanced explainability dashboard

---

# Author

Ankit
=======
