# BillWise — Medicare Provider Fraud Detection

A machine learning web app that flags potentially fraudulent healthcare providers from Medicare claims data. BillWise benchmarks a **Random Forest classifier** against an **Apriori-based, fully explainable rule decision tree** on the same held-out providers, and serves both through a **FastAPI** backend and an interactive dashboard.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen)

---

## Overview

Medicare fraud costs billions every year, and auditing every provider is impossible. BillWise turns raw claim-level records into provider-level behaviour profiles and scores each provider for fraud risk, so auditors can prioritise who to review first.

Two approaches are compared side by side:

| | Random Forest | Apriori Rule Decision Tree |
|---|---|---|
| Type | Ensemble ML (black-box) | Association-rule based (white-box) |
| Output | Fraud probability + risk level | IF-THEN rule verdict with support, confidence, lift |
| Strength | Higher precision, fewer false alarms | Higher recall, every decision is auditable |

## Dataset

Medicare provider claims data (Kaggle: *Healthcare Provider Fraud Detection Analysis*).

- **5,410** labelled providers (**506** flagged `PotentialFraud = Yes`, about 9.35%, so the data is heavily imbalanced)
- **558,211** claims in total: 40,474 inpatient and 517,737 outpatient

## How It Works

1. **Data processing** (`data_processing.py`): joins inpatient and outpatient claims and aggregates them to one row per provider.
2. **Feature engineering**: 9 provider-level features:
   - total claim count, total reimbursement, mean and max reimbursement per claim
   - inpatient and outpatient claim counts
   - unique beneficiaries, unique physicians (attending, operating, other)
   - average inpatient length of stay
3. **Provider-level stratified split** (80/20): train and test sets never share a provider, and the class ratio is preserved.
4. **Random Forest** (`model.py`): `class_weight="balanced"` to handle the imbalance.
5. **Apriori rule tree** (`apriori_tree.py`, `association_rules.py`):
   - features are discretised into Low / Medium / High using quantile edges computed **only from training data** (no leakage)
   - frequent itemsets and rules mined with `mlxtend`
   - rule conflicts resolved in strict order: **specificity, confidence, support, lift**
   - ties or unmatched inputs return **Inconclusive** rather than a forced guess
   - does not use scikit-learn's `DecisionTreeClassifier`
6. **Dual prediction API + dashboard**: both models score the same provider, and the comparison is shown with coverage so rule-based metrics are not overstated.

## Results (held-out test set: 1,082 providers, 101 fraudulent)

| Metric | Random Forest | Apriori Rule Tree |
|---|---|---|
| Accuracy | **94.55%** | 89.74% |
| Precision | **80.00%** | 46.58% |
| Recall | 55.45% | **67.33%** |
| F1-score | **65.50%** | 55.06% |
| ROC-AUC | 0.9447 | n/a |
| Coverage | 100% | 100% (0 inconclusive) |

Confusion matrices (TN / FP / FN / TP): Random Forest 967 / 14 / 45 / 56, Apriori tree 903 / 78 / 33 / 68.

**Takeaway:** Random Forest is the stronger overall model and raises far fewer false alarms. The Apriori tree catches more actual fraud cases (higher recall) and explains every verdict, which matters for regulatory auditability. Top Random Forest features: total reimbursement (about 36%), max reimbursement per claim (about 14%), inpatient claim count (about 9%).

> Numbers come from a single stratified split with `random_state=42` and the default 0.5 decision threshold. They will shift slightly with different seeds or thresholds.

## Features

- Dataset overview dashboard (class balance, claim volumes, feature descriptions)
- Model evaluation and side-by-side comparison chart, with auto-generated reasoning
- Retrain the Random Forest from the UI (number of trees, test split)
- Interactive prediction: enter billing metrics manually, or load a held-out test provider
- Provider lookup by ID with engineered features and both models' verdicts
- Browsable Apriori association rules (adjustable min support and confidence)

## Tech Stack

**Backend:** Python, FastAPI, Uvicorn, Pydantic
**ML / Data:** scikit-learn, pandas, NumPy, mlxtend (Apriori)
**Frontend:** HTML, CSS, vanilla JavaScript
**Testing:** pytest, httpx

## Project Structure

```
.
├── main.py                 # FastAPI app and REST endpoints
├── data_processing.py      # Loading, cleaning, provider-level feature engineering, split
├── model.py                # Random Forest training/eval + dual prediction engine
├── apriori_tree.py         # Apriori-based rule decision tree
├── association_rules.py    # Discretisation and association rule mining
├── test_billwise.py        # 17 tests (logic, leakage, metrics, API)
├── requirements.txt
├── static/                 # Dashboard (index.html, style.css, script.js)
└── data/                   # CSV files (see "Dataset setup")
```



### Dataset setup

Place these files in `data/` (download from the Kaggle dataset above):

```
Train-1542865627584.csv
Train_Inpatientdata-1542865627584.csv
Train_Outpatientdata-1542865627584.csv
Test-1542969243754.csv
```

The first run builds `data/processed_providers_cache.csv` automatically.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/overview` | Dataset summary |
| GET | `/api/model/evaluate` | Metrics for both models and comparison |
| POST | `/api/model/train` | Retrain Random Forest with custom parameters |
| POST | `/api/predict` | Predict fraud risk from billing metrics |
| GET | `/api/provider/{provider_id}` | Look up a provider and score it |
| GET | `/api/sample-providers` | Sample held-out providers |
| GET | `/api/test-providers` | Provider IDs from the test list |
| GET | `/api/association-rules` | Mined rules (`min_support`, `min_confidence`) |

## Running Tests

```bash
pytest -q
```

Covers rule-conflict resolution (specificity, confidence, support, lift, ties), unmatched inputs, train/test leakage, model metrics, and the API endpoints.

## Limitations and Future Work

- Fraud labels are provider-level only; there is no claim-level ground truth.
- Recall of the Random Forest is about 55%, so many fraudulent providers are still missed. Threshold tuning, SMOTE, or gradient boosting (XGBoost/LightGBM) could improve this.
- The Apriori tree trades precision for explainability and flags more legitimate providers.
- Diagnosis/procedure code features are not used yet.
- Models are retrained at startup; a persisted model (joblib) would speed up deployment.

## Acknowledgements

Dataset: Kaggle, *Healthcare Provider Fraud Detection Analysis* (Medicare claims). Use is for educational purposes.
