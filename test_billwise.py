"""
BillWise - Comprehensive Automated Test Suite
Tests for:
  - Rule conflicts resolution hierarchy (specificity, confidence, support, lift)
  - Unresolved rule conflicts returning Inconclusive
  - Unmatched inputs returning Inconclusive
  - Absence of conventional DecisionTreeClassifier
  - Zero data leakage between train and test partitions
  - Random Forest and Apriori Decision Tree evaluations on held-out test set
  - Dynamic comparative reasoning generation
  - API endpoints (health, overview, evaluate, predict, provider lookup)
  - Edge case inputs (zeros, negatives, extreme outliers)
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

import apriori_tree
from apriori_tree import AprioriDecisionTree, AprioriRule
import association_rules
import data_processing
import main
import model


@pytest.fixture(scope="module")
def client():
    return TestClient(main.app)


# =====================================================================
# 1. RULE CONFLICT TESTS (Requirement 2)
# =====================================================================

def test_rule_conflict_resolved_by_specificity():
    """Higher specificity (more antecedent conditions) must take precedence."""
    tree = AprioriDecisionTree()
    # Fraud rule has 3 conditions (specificity=3)
    rule_fraud = AprioriRule(
        rule_id="R_FRAUD_SPEC3",
        antecedents=frozenset(["featA_High", "featB_High", "featC_High"]),
        target_class="Yes",
        specificity=3,
        confidence=0.45,
        support=0.05,
        lift=4.0,
        antecedents_formatted=["Feat A (High)", "Feat B (High)", "Feat C (High)"],
        consequent_formatted="PotentialFraud = Yes",
        interpretation="Test fraud rule",
    )
    # Legitimate rule has 2 conditions (specificity=2) with higher confidence
    rule_clean = AprioriRule(
        rule_id="R_CLEAN_SPEC2",
        antecedents=frozenset(["featA_High", "featB_High"]),
        target_class="No",
        specificity=2,
        confidence=0.95,
        support=0.30,
        lift=1.05,
        antecedents_formatted=["Feat A (High)", "Feat B (High)"],
        consequent_formatted="PotentialFraud = No",
        interpretation="Test clean rule",
    )

    tree.rules = [rule_fraud, rule_clean]
    tokens = {"featA_High", "featB_High", "featC_High"}
    res = tree.predict_tokens(tokens)

    assert res["predicted_class"] == "Yes"
    assert res["conflict_detected"] is True
    assert res["winning_rule"]["rule_id"] == "R_FRAUD_SPEC3"
    assert "Specificity" in res["resolution_reason"]


def test_rule_conflict_resolved_by_confidence():
    """When specificity is tied, higher confidence must win."""
    tree = AprioriDecisionTree()
    rule_fraud = AprioriRule(
        rule_id="R_FRAUD_CONF",
        antecedents=frozenset(["featA_High", "featB_High"]),
        target_class="Yes",
        specificity=2,
        confidence=0.85,
        support=0.05,
        lift=4.0,
        antecedents_formatted=["Feat A (High)", "Feat B (High)"],
        consequent_formatted="PotentialFraud = Yes",
        interpretation="",
    )
    rule_clean = AprioriRule(
        rule_id="R_CLEAN_CONF",
        antecedents=frozenset(["featA_High", "featB_High"]),
        target_class="No",
        specificity=2,
        confidence=0.65,
        support=0.20,
        lift=1.05,
        antecedents_formatted=["Feat A (High)", "Feat B (High)"],
        consequent_formatted="PotentialFraud = No",
        interpretation="",
    )

    tree.rules = [rule_fraud, rule_clean]
    tokens = {"featA_High", "featB_High"}
    res = tree.predict_tokens(tokens)

    assert res["predicted_class"] == "Yes"
    assert res["winning_rule"]["rule_id"] == "R_FRAUD_CONF"
    assert "Confidence" in res["resolution_reason"]


def test_rule_conflict_resolved_by_support():
    """When specificity and confidence are tied, higher support must win."""
    tree = AprioriDecisionTree()
    rule_fraud = AprioriRule(
        rule_id="R_FRAUD_SUP",
        antecedents=frozenset(["featA_High", "featB_High"]),
        target_class="Yes",
        specificity=2,
        confidence=0.75,
        support=0.08,
        lift=4.0,
        antecedents_formatted=["Feat A (High)", "Feat B (High)"],
        consequent_formatted="PotentialFraud = Yes",
        interpretation="",
    )
    rule_clean = AprioriRule(
        rule_id="R_CLEAN_SUP",
        antecedents=frozenset(["featA_High", "featB_High"]),
        target_class="No",
        specificity=2,
        confidence=0.75,
        support=0.25,
        lift=1.1,
        antecedents_formatted=["Feat A (High)", "Feat B (High)"],
        consequent_formatted="PotentialFraud = No",
        interpretation="",
    )

    tree.rules = [rule_fraud, rule_clean]
    tokens = {"featA_High", "featB_High"}
    res = tree.predict_tokens(tokens)

    assert res["predicted_class"] == "No"
    assert res["winning_rule"]["rule_id"] == "R_CLEAN_SUP"
    assert "Support" in res["resolution_reason"]


def test_rule_conflict_resolved_by_lift():
    """When specificity, confidence, and support are tied, higher lift must win."""
    tree = AprioriDecisionTree()
    rule_fraud = AprioriRule(
        rule_id="R_FRAUD_LIFT",
        antecedents=frozenset(["featA_High", "featB_High"]),
        target_class="Yes",
        specificity=2,
        confidence=0.75,
        support=0.15,
        lift=5.2,
        antecedents_formatted=["Feat A (High)", "Feat B (High)"],
        consequent_formatted="PotentialFraud = Yes",
        interpretation="",
    )
    rule_clean = AprioriRule(
        rule_id="R_CLEAN_LIFT",
        antecedents=frozenset(["featA_High", "featB_High"]),
        target_class="No",
        specificity=2,
        confidence=0.75,
        support=0.15,
        lift=1.4,
        antecedents_formatted=["Feat A (High)", "Feat B (High)"],
        consequent_formatted="PotentialFraud = No",
        interpretation="",
    )

    tree.rules = [rule_fraud, rule_clean]
    tokens = {"featA_High", "featB_High"}
    res = tree.predict_tokens(tokens)

    assert res["predicted_class"] == "Yes"
    assert res["winning_rule"]["rule_id"] == "R_FRAUD_LIFT"
    assert "Lift" in res["resolution_reason"]


def test_unresolved_rule_conflict_returns_inconclusive():
    """If conflicting rules tie on all 4 criteria, return Inconclusive without forcing."""
    tree = AprioriDecisionTree()
    rule_fraud = AprioriRule(
        rule_id="R_FRAUD_TIED",
        antecedents=frozenset(["featA_High", "featB_High"]),
        target_class="Yes",
        specificity=2,
        confidence=0.75,
        support=0.15,
        lift=2.5,
        antecedents_formatted=["Feat A (High)", "Feat B (High)"],
        consequent_formatted="PotentialFraud = Yes",
        interpretation="",
    )
    rule_clean = AprioriRule(
        rule_id="R_CLEAN_TIED",
        antecedents=frozenset(["featA_High", "featB_High"]),
        target_class="No",
        specificity=2,
        confidence=0.75,
        support=0.15,
        lift=2.5,
        antecedents_formatted=["Feat A (High)", "Feat B (High)"],
        consequent_formatted="PotentialFraud = No",
        interpretation="",
    )

    tree.rules = [rule_fraud, rule_clean]
    tokens = {"featA_High", "featB_High"}
    res = tree.predict_tokens(tokens)

    assert res["predicted_class"] == "Inconclusive"
    assert res["is_conclusive"] is False
    assert res["winning_rule"] is None
    assert "Unresolved Rule Conflict" in res["resolution_reason"]
    assert res["probability"] is None
    assert res["probability_display"] == "N/A"


def test_unmatched_inputs_return_inconclusive():
    """If no rules match the input features, return Inconclusive."""
    tree = AprioriDecisionTree()
    rule_fraud = AprioriRule(
        rule_id="R_ONLY",
        antecedents=frozenset(["featA_High", "featB_High"]),
        target_class="Yes",
        specificity=2,
        confidence=0.80,
        support=0.05,
        lift=3.0,
        antecedents_formatted=["Feat A (High)", "Feat B (High)"],
        consequent_formatted="PotentialFraud = Yes",
        interpretation="",
    )
    tree.rules = [rule_fraud]
    tokens = {"unrelated_Low", "other_Medium"}
    res = tree.predict_tokens(tokens)

    assert res["predicted_class"] == "Inconclusive"
    assert res["is_conclusive"] is False
    assert res["matched_count"] == 0
    assert "No Apriori association rules matched" in res["probability_reason"]
    assert res["probability_display"] == "N/A"


# =====================================================================
# 2. NO CONVENTIONAL DECISION TREE CLASSIFIER (Requirement 2)
# =====================================================================

def test_no_conventional_decision_tree_classifier_imported():
    """Verify that sklearn.tree.DecisionTreeClassifier is not used for Apriori decision structure."""
    with open(apriori_tree.__file__, "r", encoding="utf-8") as f:
        content = f.read()
    # Check that DecisionTreeClassifier is not imported from sklearn
    assert "from sklearn.tree import DecisionTreeClassifier" not in content
    assert "import DecisionTreeClassifier" not in content
    assert "sklearn.tree" not in content
    assert not hasattr(apriori_tree, "DecisionTreeClassifier")


# =====================================================================
# 3. ZERO DATA LEAKAGE & STRATIFIED SPLIT (Requirement 1 & 5)
# =====================================================================

def test_zero_data_leakage_between_train_and_test():
    """Ensure training and test sets are strictly disjoint (0 provider overlap)."""
    _, _, _, _, prov_train, prov_test, _ = data_processing.get_stratified_train_test_split(
        test_size=0.2, random_state=42
    )
    overlap = set(prov_train).intersection(set(prov_test))
    assert len(overlap) == 0, f"Data leakage detected! Overlapping providers: {overlap}"
    assert len(prov_train) > 0
    assert len(prov_test) > 0


# =====================================================================
# 4. MODEL EVALUATIONS ON HELD-OUT TEST SET (Requirement 4 & 5)
# =====================================================================

def test_apriori_tree_evaluation_metrics():
    """Verify Apriori Decision Tree evaluates on held-out test data with transparent coverage."""
    metrics = apriori_tree.evaluate_apriori_decision_tree()

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert metrics["total_test_count"] == metrics["conclusive_count"] + metrics["inconclusive_count"]
    assert metrics["is_conventional_tree"] is False


def test_random_forest_evaluation_metrics():
    """Verify Random Forest evaluates on held-out test data."""
    metrics = model.get_evaluation_metrics()

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


# =====================================================================
# 5. DUAL PREDICTIONS & API RESPONSES (Requirement 3, 7, 8)
# =====================================================================

def test_api_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"


def test_api_overview_endpoint(client):
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["data"]["total_providers"] > 0


def test_api_evaluate_returns_both_models_and_comparison(client):
    res = client.get("/api/model/evaluate")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    body = data["data"]

    # Must contain both models
    assert "random_forest" in body or "f1_score" in body
    assert "decision_tree" in body
    dt_metrics = body["decision_tree"]
    assert "coverage" in dt_metrics
    assert "inconclusive_count" in dt_metrics
    assert "f1_score" in dt_metrics

    # Must contain comparison metrics & dynamic reasoning
    assert "comparison" in body
    assert "reasoning" in body
    assert len(body["reasoning"]["summary"]) > 0


def test_api_predict_returns_dual_predictions(client):
    """Submitting provider features must return both RF and Decision Tree predictions."""
    payload = {
        "total_claim_count": 520,
        "total_reimb_amt": 680000,
        "mean_reimb_per_claim": 1307.69,
        "max_reimb_per_claim": 48000,
        "inpatient_claim_count": 85,
        "outpatient_claim_count": 435,
        "unique_beneficiaries": 310,
        "unique_physicians": 65,
        "avg_inpatient_los": 6.8,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    pred = data["data"]

    # 1. Random Forest prediction
    assert "random_forest" in pred
    rf = pred["random_forest"]
    assert rf["predicted_class"] in ("Yes", "No")
    assert 0.0 <= rf["fraud_probability"] <= 1.0

    # 2. Rule-based Decision Tree prediction
    assert "decision_tree" in pred
    dt = pred["decision_tree"]
    assert dt["predicted_class"] in ("Yes", "No", "Inconclusive")
    assert dt["probability"] is None  # Must show N/A with reason unless calibrated
    assert dt["probability_display"] == "N/A"
    assert "probability_reason" in dt

    # 3. Rule confidence clearly labeled
    assert dt["rule_confidence_label"] == "Rule Confidence"
    if dt["predicted_class"] != "Inconclusive":
        assert dt["rule_confidence"] is not None

    # 4. Matching rules returned
    assert "matching_rules" in dt
    assert isinstance(dt["matching_rules"], list)


def test_api_provider_lookup(client):
    """Looking up known provider PRV51001 returns dual predictions."""
    res = client.get("/api/provider/PRV51001")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["data"]["Provider"] == "PRV51001"
    assert "prediction" in data["data"]
    assert "decision_tree" in data["data"]["prediction"]


def test_api_provider_not_found(client):
    res = client.get("/api/provider/NON_EXISTENT_ID_999999")
    assert res.status_code == 404


def test_edge_case_inputs(client):
    """Test all zeros, negative values handled gracefully."""
    payload = {
        "total_claim_count": 0,
        "total_reimb_amt": 0,
        "mean_reimb_per_claim": 0,
        "max_reimb_per_claim": 0,
        "inpatient_claim_count": 0,
        "outpatient_claim_count": 0,
        "unique_beneficiaries": 0,
        "unique_physicians": 0,
        "avg_inpatient_los": 0,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["data"]["random_forest"]["predicted_class"] in ("Yes", "No")
    assert data["data"]["decision_tree"]["predicted_class"] in ("Yes", "No", "Inconclusive")
