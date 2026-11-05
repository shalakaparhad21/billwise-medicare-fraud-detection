# BillWise – Healthcare Provider Billing Fraud Detection System

An end-to-end Machine Learning application that identifies potentially fraudulent healthcare providers by analyzing Medicare inpatient and outpatient billing patterns.

## Overview

BillWise analyzes **558K Medicare claims across 5,410 healthcare providers** to detect suspicious billing patterns. It combines supervised machine learning with explainable association rule mining to predict potentially fraudulent providers and provide interpretable insights.

## Key Features

- Processes inpatient and outpatient Medicare claims.
- Engineers 9 provider-level features from raw claim data.
- Uses a class-balanced Random Forest classifier for fraud prediction.
- Implements an explainable Apriori-based rule decision tree.
- Compares both models using a common stratified train-test split.
- Provides provider-level predictions, fraud probabilities, and matched rule explanations through an interactive dashboard.

## Machine Learning Models

| Model | Performance |
|---|---|
| Random Forest | 94.6% Accuracy, 0.94 ROC-AUC |
| Apriori-Based Rule Decision Tree | 67% Recall |

**Random Forest:** Uses 100 decision trees with balanced class weights to handle class imbalance.

**Apriori-Based Rule Decision Tree:** Mines association rules using support, confidence, and lift to generate interpretable, rule-based predictions.

## Feature Engineering

Nine provider-level features are extracted:

- Total claim count
- Total reimbursement amount
- Mean reimbursement per claim
- Maximum reimbursement per claim
- Inpatient claim count
- Outpatient claim count
- Unique beneficiaries
- Unique physicians
- Average inpatient length of stay

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Association Rule Mining:** MLxtend
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Testing:** Pytest, HTTPX

## Project Structure

```text
BillWise/
├── data/
├── static/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── data_processing.py
├── model.py
├── association_rules.py
├── apriori_tree.py
├── main.py
├── test_billwise.py
├── requirements.txt
└── README.md
```

**BillWise – Combining Machine Learning and Explainable AI for Healthcare Fraud Detection.**
