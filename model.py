"""
BillWise - Model Evaluation & Dual Prediction Engine
Handles:
  1. Random Forest Classifier training and held-out test evaluation.
  2. Apriori-based rule Decision Tree evaluation on the identical test split.
  3. Dynamic comparative reasoning and metric benchmarking (Accuracy, Precision, Recall, F1).
  4. Real-time dual predictions combining Random Forest risk scoring and Apriori rule decisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import apriori_tree
from data_processing import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    get_stratified_train_test_split,
)

_TRAINED_MODEL = None
_MODEL_METRICS = None
_TEST_DATA = None


def generate_comparative_reasoning(rf_metrics: dict, dt_metrics: dict) -> dict:
    """
    Dynamically generates objective, metrics-based reasoning comparing Random Forest
    and the Apriori-based Decision Tree evaluated on the held-out test set.
    Highlights accuracy, precision, recall, F1, interpretability, and decision coverage.
    """
    rf_f1 = rf_metrics["f1_score"]
    dt_f1 = dt_metrics["f1_score"]
    rf_prec = rf_metrics["precision"]
    dt_prec = dt_metrics["precision"]
    rf_rec = rf_metrics["recall"]
    dt_rec = dt_metrics["recall"]
    rf_acc = rf_metrics["accuracy"]
    dt_acc = dt_metrics["accuracy"]

    coverage_pct = dt_metrics["coverage_pct"]
    inconclusive_count = dt_metrics["inconclusive_count"]
    total_test = dt_metrics["total_test_count"]
    conclusive_count = dt_metrics["conclusive_count"]

    # Determine better performing model on primary balanced metric (F1-score)
    f1_diff = rf_f1 - dt_f1
    if abs(f1_diff) < 0.01:
        better_model = "Comparable Overall Performance"
        better_badge = "Tie / Trade-off"
        summary = (
            f"Both models demonstrate comparable overall predictive balance on held-out test providers "
            f"(Random Forest F1: {rf_f1 * 100:.1f}%, Apriori Tree F1: {dt_f1 * 100:.1f}%)."
        )
    elif f1_diff > 0:
        better_model = "Random Forest Classifier"
        better_badge = "Superior Overall F1"
        summary = (
            f"Random Forest outperforms the Apriori-based Decision Tree in overall balanced performance "
            f"with an F1-score of {rf_f1 * 100:.1f}% vs. {dt_f1 * 100:.1f}% (+{f1_diff * 100:.1f}%). "
            f"Its continuous probabilistic ensemble reduces false positives, achieving {rf_prec * 100:.1f}% precision."
        )
    else:
        better_model = "Apriori-based Decision Tree"
        better_badge = "Superior Overall F1"
        summary = (
            f"The Apriori-based Decision Tree achieves a higher overall F1-score ({dt_f1 * 100:.1f}% vs. {rf_f1 * 100:.1f}%), "
            f"leveraging highly targeted frequent fraud billing itemsets."
        )

    # Detailed trade-off points
    points = []

    # Precision analysis
    if rf_prec > dt_prec:
        points.append({
            "dimension": "Audit Precision (False Alarm Reduction)",
            "leader": "Random Forest",
            "text": (
                f"Random Forest achieves higher precision ({rf_prec * 100:.1f}% vs. {dt_prec * 100:.1f}%), "
                f"meaning fewer non-fraudulent providers are flagged unnecessarily for expensive Medicare audits."
            ),
        })
    else:
        points.append({
            "dimension": "Audit Precision",
            "leader": "Apriori Decision Tree",
            "text": (
                f"Apriori Decision Tree delivers higher precision ({dt_prec * 100:.1f}% vs. {rf_prec * 100:.1f}%), "
                f"ensuring flags are backed by specific high-confidence rule combinations."
            ),
        })

    # Recall analysis
    if rf_rec > dt_rec:
        points.append({
            "dimension": "Fraud Recall (Sensitivity)",
            "leader": "Random Forest",
            "text": (
                f"Random Forest catches a higher proportion of fraudulent providers ({rf_rec * 100:.1f}% vs. {dt_rec * 100:.1f}%), "
                f"reducing missed billing anomalies."
            ),
        })
    else:
        points.append({
            "dimension": "Fraud Recall (Sensitivity)",
            "leader": "Apriori Decision Tree",
            "text": (
                f"Apriori Decision Tree captures {dt_rec * 100:.1f}% of actual fraud cases (vs. {rf_rec * 100:.1f}% for RF), "
                f"proving that high-lift billing combinations effectively isolate fraudulent billing behavior."
            ),
        })

    # Interpretability & Transparency analysis
    points.append({
        "dimension": "Explainability & Regulatory Auditability",
        "leader": "Apriori Decision Tree",
        "text": (
            "While Random Forest provides continuous probability scoring, the Apriori Decision Tree is fully "
            "white-box: every verdict maps to explicit IF-THEN rules with verified support, confidence, and lift."
        ),
    })

    # Coverage transparency note (Requirement 5)
    coverage_explanation = (
        f"The Apriori Decision Tree evaluated {conclusive_count}/{total_test} held-out test providers "
        f"({coverage_pct}% coverage) with {inconclusive_count} inconclusive predictions. "
        f"Because rule-based metrics apply strictly to conclusive decisions, presenting coverage alongside "
        f"accuracy prevents performance overestimation."
    )

    return {
        "better_model": better_model,
        "better_badge": better_badge,
        "summary": summary,
        "comparison_points": points,
        "coverage_explanation": coverage_explanation,
    }


def train_and_evaluate_model(
    n_estimators: int = 100,
    max_depth: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Trains Random Forest Classifier on provider training partition with class_weight='balanced'.
    Simultaneously evaluates the Apriori Decision Tree on the exact same held-out test providers
    to ensure 100% zero data leakage and fair benchmarking.
    """
    global _TRAINED_MODEL, _MODEL_METRICS, _TEST_DATA

    X_train, X_test, y_train, y_test, prov_train, prov_test, full_df = get_stratified_train_test_split(
        test_size=test_size, random_state=random_state
    )

    # 1. Fit Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # RF held-out evaluation
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))

    # Feature importance
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feature_importance_list = [
        {
            "feature": FEATURE_COLUMNS[i],
            "label": FEATURE_LABELS[FEATURE_COLUMNS[i]],
            "importance": round(float(importances[i]), 4),
            "percentage": round(float(importances[i] * 100), 2),
        }
        for i in sorted_idx
    ]

    rf_metrics = {
        "model_name": "Random Forest Classifier",
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "roc_auc": round(auc, 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "feature_importance": feature_importance_list,
    }

    # 2. Evaluate Apriori Decision Tree on the exact same test partition
    test_df_slice = full_df[full_df["Provider"].isin(prov_test)].copy()
    dt = apriori_tree.get_apriori_decision_tree(force_retrain=False)
    dt_metrics = dt.evaluate_on_test_set(test_df_slice)

    # 3. Dynamic comparative reasoning
    reasoning = generate_comparative_reasoning(rf_metrics, dt_metrics)

    # 4. Comparison summary data structure for graphs
    comparison_data = {
        "metrics": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "random_forest": [
            round(rf_metrics["accuracy"] * 100, 2),
            round(rf_metrics["precision"] * 100, 2),
            round(rf_metrics["recall"] * 100, 2),
            round(rf_metrics["f1_score"] * 100, 2),
        ],
        "decision_tree": [
            round(dt_metrics["accuracy"] * 100, 2),
            round(dt_metrics["precision"] * 100, 2),
            round(dt_metrics["recall"] * 100, 2),
            round(dt_metrics["f1_score"] * 100, 2),
        ],
        "coverage": {
            "random_forest": 100.0,
            "decision_tree": dt_metrics["coverage_pct"],
            "decision_tree_inconclusive": dt_metrics["inconclusive_count"],
            "decision_tree_conclusive": dt_metrics["conclusive_count"],
            "total_test": len(X_test),
        },
    }

    _TRAINED_MODEL = rf

    # Retain all top-level keys for backwards compatibility, plus rich models object
    _MODEL_METRICS = {
        **rf_metrics,
        "random_forest": rf_metrics,
        "decision_tree": dt_metrics,
        "comparison": comparison_data,
        "reasoning": reasoning,
    }

    # Store test providers with ground truth and features for interactive lookup
    test_records = []
    test_df_slice["RF_Predicted"] = ["Yes" if p == 1 else "No" for p in y_pred]
    test_df_slice["RF_Probability"] = [round(float(p), 4) for p in y_prob]

    for _, row in test_df_slice.iterrows():
        feat = {col: float(row[col]) for col in FEATURE_COLUMNS}
        dt_res = dt.predict_provider(feat, top_n_rules=1)
        rec = {
            "Provider": row["Provider"],
            "ActualFraud": row["PotentialFraud"],
            "PredictedFraud": row["RF_Predicted"],
            "Probability": row["RF_Probability"],
            "DT_Predicted": dt_res["predicted_class"],
            "DT_RuleConfidence": dt_res["rule_confidence_pct"],
            "features": feat,
        }
        test_records.append(rec)

    _TEST_DATA = test_records

    return _MODEL_METRICS


def get_model():
    """Returns trained Random Forest model."""
    global _TRAINED_MODEL
    if _TRAINED_MODEL is None:
        train_and_evaluate_model()
    return _TRAINED_MODEL


def get_evaluation_metrics() -> dict:
    """Returns model metrics for both Random Forest and Apriori Decision Tree."""
    global _MODEL_METRICS
    if _MODEL_METRICS is None:
        train_and_evaluate_model()
    return _MODEL_METRICS


def get_sample_test_providers(limit: int = 10) -> list:
    """Returns sample held-out test providers with actual labels and dual predictions."""
    global _TEST_DATA
    if _TEST_DATA is None:
        train_and_evaluate_model()

    fraud_samples = [r for r in _TEST_DATA if r["ActualFraud"] == "Yes"][:5]
    non_fraud_samples = [r for r in _TEST_DATA if r["ActualFraud"] == "No"][:5]
    return fraud_samples + non_fraud_samples


def predict_provider_fraud(feature_dict: dict) -> dict:
    """
    Dual prediction engine:
    Computes both Random Forest risk probability/class AND Apriori-based Decision Tree prediction.
    Explicitly labels Apriori confidence as Rule Confidence (not fraud probability).
    Displays Decision Tree probability as N/A with reason unless calibrated.
    """
    rf_model = get_model()

    # Build input row in exact feature order
    row_values = []
    for col in FEATURE_COLUMNS:
        val = feature_dict.get(col, 0.0)
        try:
            row_values.append(float(val))
        except (ValueError, TypeError):
            row_values.append(0.0)

    X_input = pd.DataFrame([row_values], columns=FEATURE_COLUMNS)

    # 1. Random Forest prediction
    pred_class_idx = int(rf_model.predict(X_input)[0])
    probabilities = rf_model.predict_proba(X_input)[0]
    fraud_prob = float(probabilities[1])

    predicted_label = "Yes" if pred_class_idx == 1 else "No"
    status_text = "Potential Fraud Flagged" if pred_class_idx == 1 else "Normal / Legitimate Provider"

    if fraud_prob >= 0.70:
        risk_level = "High Risk"
        risk_color = "danger"
    elif fraud_prob >= 0.40:
        risk_level = "Moderate Risk"
        risk_color = "warning"
    else:
        risk_level = "Low Risk"
        risk_color = "success"

    importances = dict(zip(FEATURE_COLUMNS, rf_model.feature_importances_))
    top_features = sorted(FEATURE_COLUMNS, key=lambda f: importances[f], reverse=True)[:3]
    contributing_factors = [
        {
            "feature": feat,
            "label": FEATURE_LABELS[feat],
            "value": feature_dict.get(feat, 0.0),
            "importance": round(float(importances[feat]), 4),
        }
        for feat in top_features
    ]

    rf_result = {
        "model_name": "Random Forest Classifier",
        "predicted_class": predicted_label,
        "status_text": status_text,
        "fraud_probability": round(fraud_prob, 4),
        "fraud_percentage": round(fraud_prob * 100, 2),
        "non_fraud_probability": round(float(probabilities[0]), 4),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "contributing_factors": contributing_factors,
    }

    # 2. Apriori Decision Tree prediction
    dt_result = apriori_tree.predict_with_apriori_tree(feature_dict)

    # Combined dual response
    return {
        # Top-level legacy keys preserved
        "predicted_class": predicted_label,
        "status_text": status_text,
        "fraud_probability": round(fraud_prob, 4),
        "fraud_percentage": round(fraud_prob * 100, 2),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "contributing_factors": contributing_factors,
        "input_features": feature_dict,
        # Dedicated dual prediction structures
        "random_forest": rf_result,
        "decision_tree": dt_result,
    }
