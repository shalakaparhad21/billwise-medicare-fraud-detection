"""
BillWise - Apriori-Based Decision Tree Module
Implements a rule-based decision structure using ONLY Apriori-generated association rules.
Does NOT train or use a conventional DecisionTreeClassifier (scikit-learn CART/C4.5).

Handles rule conflicts strictly using the user-specified hierarchical precedence:
  1. Specificity (number of antecedent conditions: more specific rules take precedence)
  2. Confidence (rule conditional precision)
  3. Support (rule prevalence in training data)
  4. Lift (relative increase over baseline)

If conflicts remain unresolved (tied on all 4 criteria) or if no rules match the input,
returns an Inconclusive prediction without forcing a default classification.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules as mlx_rules
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data_processing import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    get_stratified_train_test_split,
)
import association_rules


class AprioriRule:
    """Represents a single classification rule mined by Apriori."""

    def __init__(
        self,
        rule_id: str,
        antecedents: frozenset[str],
        target_class: str,
        specificity: int,
        confidence: float,
        support: float,
        lift: float,
        antecedents_formatted: list[str],
        consequent_formatted: str,
        interpretation: str,
    ):
        self.rule_id = rule_id
        self.antecedents = antecedents
        self.target_class = target_class  # 'Yes' or 'No'
        self.specificity = specificity
        self.confidence = confidence
        self.support = support
        self.lift = lift
        self.antecedents_formatted = antecedents_formatted
        self.consequent_formatted = consequent_formatted
        self.interpretation = interpretation

    def matches(self, provider_tokens: Set[str]) -> bool:
        """Rule fires if all antecedent conditions are present in the provider's tokens."""
        return self.antecedents.issubset(provider_tokens)

    def sort_tuple(self) -> Tuple[int, float, float, float]:
        """Strict conflict resolution tuple: specificity, confidence, support, lift."""
        return (self.specificity, self.confidence, self.support, self.lift)

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "antecedents": self.antecedents_formatted,
            "antecedent_tokens": sorted(list(self.antecedents)),
            "target_class": self.target_class,
            "consequent": self.consequent_formatted,
            "specificity": self.specificity,
            "confidence": round(self.confidence, 4),
            "confidence_pct": round(self.confidence * 100, 2),
            "support": round(self.support, 4),
            "support_pct": round(self.support * 100, 2),
            "lift": round(self.lift, 3),
            "interpretation": self.interpretation,
        }


class AprioriTreeNode:
    """
    Hierarchical decision structure node representing an Apriori condition branch.
    Forms a prefix-tree structure over Apriori itemset antecedents.
    """

    def __init__(self, condition: str = "root", depth: int = 0):
        self.condition = condition
        self.depth = depth
        self.children: Dict[str, AprioriTreeNode] = {}
        self.leaf_rules: List[AprioriRule] = []

    def insert_rule(self, remaining_tokens: list[str], rule: AprioriRule):
        if not remaining_tokens:
            self.leaf_rules.append(rule)
            return
        token = remaining_tokens[0]
        if token not in self.children:
            self.children[token] = AprioriTreeNode(condition=token, depth=self.depth + 1)
        self.children[token].insert_rule(remaining_tokens[1:], rule)

    def to_summary_dict(self, max_depth: int = 2) -> dict:
        """Summarize top levels of the decision tree for introspection."""
        summary = {
            "condition": self.condition,
            "depth": self.depth,
            "rules_count": len(self.leaf_rules),
            "children_count": len(self.children),
        }
        if self.depth < max_depth:
            summary["branches"] = {
                k: v.to_summary_dict(max_depth=max_depth)
                for k, v in list(self.children.items())[:5]
            }
        return summary


class AprioriDecisionTree:
    """
    Apriori-based rule Decision Tree.
    Built strictly from frequent itemsets and association rules mined on training data.
    Does NOT use scikit-learn DecisionTreeClassifier.
    """

    def __init__(
        self,
        min_support: float = 0.03,
        min_fraud_confidence: float = 0.20,
        min_clean_confidence: float = 0.70,
        max_len: int = 4,
        random_state: int = 42,
    ):
        self.min_support = min_support
        self.min_fraud_confidence = min_fraud_confidence
        self.min_clean_confidence = min_clean_confidence
        self.max_len = max_len
        self.random_state = random_state

        self.rules: List[AprioriRule] = []
        self.bin_edges: Optional[dict] = None
        self.base_fraud_rate: float = 0.0
        self.train_size: int = 0
        self.tree_root: AprioriTreeNode = AprioriTreeNode(condition="root", depth=0)
        self._metrics_cache: Optional[dict] = None

    def fit_from_training_data(self, force_retrain: bool = False):
        """Mines rules strictly from training data and builds decision structure."""
        if self.rules and not force_retrain:
            return self

        X_train, _, _, _, prov_train, _, full_df = get_stratified_train_test_split(
            test_size=0.2, random_state=self.random_state
        )
        train_df = full_df[full_df["Provider"].isin(prov_train)].copy()
        self.train_size = len(train_df)
        self.base_fraud_rate = float((train_df["PotentialFraud"] == "Yes").mean())

        # Quantile bin edges computed strictly from training providers (zero leakage)
        self.bin_edges = association_rules._compute_bin_edges(train_df)
        one_hot = association_rules._build_one_hot_frame(train_df, bin_edges=self.bin_edges)

        # Mine frequent itemsets
        frequent = apriori(
            one_hot,
            min_support=self.min_support,
            use_colnames=True,
            max_len=self.max_len,
            low_memory=True,
        )

        if frequent.empty:
            self.rules = []
            return self

        # Mine all association rules meeting the lower confidence threshold
        try:
            rules_df = mlx_rules(
                frequent,
                metric="confidence",
                min_threshold=min(self.min_fraud_confidence, self.min_clean_confidence),
                num_itemsets=len(frequent),
            )
        except TypeError:
            rules_df = mlx_rules(
                frequent,
                metric="confidence",
                min_threshold=min(self.min_fraud_confidence, self.min_clean_confidence),
            )

        if rules_df.empty:
            self.rules = []
            return self

        # Helper to classify consequent
        def parse_consequent(c):
            s = set(c)
            if s == {"PotentialFraud=Yes"}:
                return "Yes"
            if s == {"PotentialFraud=No"}:
                return "No"
            return None

        rules_df["target_class"] = rules_df["consequents"].apply(parse_consequent)
        class_rules_df = rules_df[rules_df["target_class"].notna()].copy()

        # Enforce minimum thresholds per class
        # For fraud: confidence >= min_fraud_confidence, lift > 1.0 (positive association)
        # For non-fraud: confidence >= min_clean_confidence, lift > 1.0
        fraud_mask = (class_rules_df["target_class"] == "Yes") & (
            class_rules_df["confidence"] >= self.min_fraud_confidence
        ) & (class_rules_df["lift"] > 1.0)

        clean_mask = (class_rules_df["target_class"] == "No") & (
            class_rules_df["confidence"] >= self.min_clean_confidence
        ) & (class_rules_df["lift"] > 1.0)

        filtered_df = class_rules_df[fraud_mask | clean_mask].copy()

        parsed_rules: List[AprioriRule] = []
        self.tree_root = AprioriTreeNode(condition="root", depth=0)

        rule_idx = 1
        for _, row in filtered_df.iterrows():
            antecedents_set = frozenset([
                a for a in row["antecedents"]
                if not str(a).startswith("PotentialFraud")
            ])
            if not antecedents_set:
                continue

            target_class = str(row["target_class"])
            conf = float(row["confidence"])
            sup = float(row["support"])
            lift_val = float(row["lift"])
            spec = len(antecedents_set)

            ant_formatted = [association_rules._format_item(a) for a in sorted(antecedents_set)]
            target_formatted = (
                "PotentialFraud = Yes" if target_class == "Yes" else "PotentialFraud = No"
            )

            pattern = " AND ".join(ant_formatted)
            interpretation = (
                f"When {pattern}, {target_class.lower() == 'yes' and 'fraud likelihood' or 'legitimate likelihood'} "
                f"is {conf * 100:.1f}% (lift {lift_val:.2f}x vs baseline, support {sup * 100:.1f}%)."
            )

            rule_obj = AprioriRule(
                rule_id=f"R{rule_idx}",
                antecedents=antecedents_set,
                target_class=target_class,
                specificity=spec,
                confidence=conf,
                support=sup,
                lift=lift_val,
                antecedents_formatted=ant_formatted,
                consequent_formatted=target_formatted,
                interpretation=interpretation,
            )
            parsed_rules.append(rule_obj)
            rule_idx += 1

            # Insert into tree structure (sorted tokens for canonical path)
            sorted_tokens = sorted(list(antecedents_set))
            self.tree_root.insert_rule(sorted_tokens, rule_obj)

        self.rules = parsed_rules
        return self

    def discretize_features(self, feature_dict: dict) -> Set[str]:
        """Map numeric feature inputs into Apriori discrete tokens using training bin edges."""
        if self.bin_edges is None:
            self.fit_from_training_data()

        tokens = set()
        for col in FEATURE_COLUMNS:
            try:
                val = float(feature_dict.get(col, 0.0) or 0.0)
            except (ValueError, TypeError):
                val = 0.0
            bins = self.bin_edges.get(col) if self.bin_edges else None
            series_binned = association_rules._discretize_series(pd.Series([val]), col, bins=bins)
            tokens.add(str(series_binned.iloc[0]))
        return tokens

    def predict_tokens(self, provider_tokens: Set[str], top_n_rules: int = 5) -> dict:
        """
        Evaluates input tokens against the Apriori decision structure.
        Applies strict conflict resolution: Specificity -> Confidence -> Support -> Lift.
        Returns Inconclusive if no rules match or conflicts remain unresolved.
        """
        if not self.rules:
            self.fit_from_training_data()

        # Find matching rules
        matched_rules = [r for r in self.rules if r.matches(provider_tokens)]

        if not matched_rules:
            return {
                "predicted_class": "Inconclusive",
                "status_text": "Inconclusive (No Rules Matched)",
                "status_color": "warning",
                "is_conclusive": False,
                "probability": None,
                "probability_display": "N/A",
                "probability_reason": (
                    "No Apriori association rules matched this provider's discrete billing pattern. "
                    "An inconclusive result is returned instead of forcing a default prediction."
                ),
                "rule_confidence": None,
                "rule_confidence_pct": None,
                "rule_confidence_display": "N/A",
                "winning_rule": None,
                "matched_count": 0,
                "matched_rules": [],
                "matching_rules": [],
                "resolution_reason": (
                    "Unmatched input: Provider's Low/Medium/High metrics did not satisfy the antecedent "
                    "of any mined association rule."
                ),
                "conflict_detected": False,
                "provider_tokens": sorted(list(provider_tokens)),
            }

        yes_rules = [r for r in matched_rules if r.target_class == "Yes"]
        no_rules = [r for r in matched_rules if r.target_class == "No"]

        # Sort within class by (specificity desc, confidence desc, support desc, lift desc)
        yes_rules.sort(key=lambda r: r.sort_tuple(), reverse=True)
        no_rules.sort(key=lambda r: r.sort_tuple(), reverse=True)

        conflict_detected = bool(yes_rules and no_rules)
        winning_rule: Optional[AprioriRule] = None
        predicted_class: str = "Inconclusive"
        resolution_reason: str = ""

        if yes_rules and not no_rules:
            winning_rule = yes_rules[0]
            predicted_class = "Yes"
            resolution_reason = (
                f"Unanimous match: {len(yes_rules)} fraud rule(s) fired with 0 opposing rules. "
                f"Winning rule {winning_rule.rule_id} has specificity {winning_rule.specificity} "
                f"and rule confidence {winning_rule.confidence * 100:.1f}%."
            )
        elif no_rules and not yes_rules:
            winning_rule = no_rules[0]
            predicted_class = "No"
            resolution_reason = (
                f"Unanimous match: {len(no_rules)} non-fraud rule(s) fired with 0 opposing rules. "
                f"Winning rule {winning_rule.rule_id} has specificity {winning_rule.specificity} "
                f"and rule confidence {winning_rule.confidence * 100:.1f}%."
            )
        else:
            # Both classes fired! Resolve conflict strictly:
            # 1. Specificity -> 2. Confidence -> 3. Support -> 4. Lift
            best_yes = yes_rules[0]
            best_no = no_rules[0]

            # 1. Specificity check
            if best_yes.specificity > best_no.specificity:
                winning_rule = best_yes
                predicted_class = "Yes"
                resolution_reason = (
                    f"Conflict resolved by Specificity: Top Fraud rule ({best_yes.rule_id}) has "
                    f"{best_yes.specificity} conditions vs {best_no.specificity} conditions for Non-Fraud ({best_no.rule_id})."
                )
            elif best_no.specificity > best_yes.specificity:
                winning_rule = best_no
                predicted_class = "No"
                resolution_reason = (
                    f"Conflict resolved by Specificity: Top Non-Fraud rule ({best_no.rule_id}) has "
                    f"{best_no.specificity} conditions vs {best_yes.specificity} conditions for Fraud ({best_yes.rule_id})."
                )
            else:
                # 2. Confidence check (specificity tied)
                eps = 1e-6
                conf_diff = best_yes.confidence - best_no.confidence
                if conf_diff > eps:
                    winning_rule = best_yes
                    predicted_class = "Yes"
                    resolution_reason = (
                        f"Conflict resolved by Rule Confidence: Equal specificity ({best_yes.specificity}), "
                        f"Fraud rule has {best_yes.confidence * 100:.1f}% vs Non-Fraud {best_no.confidence * 100:.1f}%."
                    )
                elif conf_diff < -eps:
                    winning_rule = best_no
                    predicted_class = "No"
                    resolution_reason = (
                        f"Conflict resolved by Rule Confidence: Equal specificity ({best_no.specificity}), "
                        f"Non-Fraud rule has {best_no.confidence * 100:.1f}% vs Fraud {best_yes.confidence * 100:.1f}%."
                    )
                else:
                    # 3. Support check (specificity and confidence tied)
                    sup_diff = best_yes.support - best_no.support
                    if sup_diff > eps:
                        winning_rule = best_yes
                        predicted_class = "Yes"
                        resolution_reason = (
                            f"Conflict resolved by Support: Tied on specificity and confidence, "
                            f"Fraud rule has higher support ({best_yes.support * 100:.2f}% vs {best_no.support * 100:.2f}%)."
                        )
                    elif sup_diff < -eps:
                        winning_rule = best_no
                        predicted_class = "No"
                        resolution_reason = (
                            f"Conflict resolved by Support: Tied on specificity and confidence, "
                            f"Non-Fraud rule has higher support ({best_no.support * 100:.2f}% vs {best_yes.support * 100:.2f}%)."
                        )
                    else:
                        # 4. Lift check (specificity, confidence, support tied)
                        lift_diff = best_yes.lift - best_no.lift
                        if lift_diff > eps:
                            winning_rule = best_yes
                            predicted_class = "Yes"
                            resolution_reason = (
                                f"Conflict resolved by Lift: Tied on specificity, confidence, and support, "
                                f"Fraud rule has higher lift ({best_yes.lift:.3f}x vs {best_no.lift:.3f}x)."
                            )
                        elif lift_diff < -eps:
                            winning_rule = best_no
                            predicted_class = "No"
                            resolution_reason = (
                                f"Conflict resolved by Lift: Tied on specificity, confidence, and support, "
                                f"Non-Fraud rule has higher lift ({best_no.lift:.3f}x vs {best_yes.lift:.3f}x)."
                            )
                        else:
                            # 5. ALL 4 TIED -> UNRESOLVED CONFLICT -> INCONCLUSIVE
                            winning_rule = None
                            predicted_class = "Inconclusive"
                            resolution_reason = (
                                f"Unresolved Rule Conflict: Fraud rule ({best_yes.rule_id}) and Non-Fraud rule ({best_no.rule_id}) "
                                f"tied identically across all 4 conflict resolution criteria (Specificity: {best_yes.specificity}, "
                                f"Confidence: {best_yes.confidence * 100:.1f}%, Support: {best_yes.support * 100:.2f}%, "
                                f"Lift: {best_yes.lift:.3f}x). Neither rule dominates."
                            )

        # Build output response
        is_conclusive = predicted_class in ("Yes", "No")
        status_text = (
            "Potential Fraud Flagged"
            if predicted_class == "Yes"
            else ("Normal / Legitimate Provider" if predicted_class == "No" else "Inconclusive Prediction")
        )
        status_color = "danger" if predicted_class == "Yes" else ("success" if predicted_class == "No" else "warning")

        # Probability handling as explicitly required:
        # "Display a Decision Tree probability only if it is defensibly estimated and calibrated; otherwise show N/A with reason."
        # Apriori decision tree outputs discrete categorical decisions; rule confidence is subset precision, not calibrated posterior probability.
        prob_val = None
        prob_display = "N/A"
        prob_reason = (
            "Apriori Decision Tree evaluates discrete association rules without a continuous "
            "calibrated posterior probability estimator. Rule confidence indicates subset precision, "
            "not overall provider fraud probability."
        )

        rule_conf = winning_rule.confidence if winning_rule else None
        rule_conf_pct = round(rule_conf * 100, 2) if rule_conf is not None else None
        rule_conf_display = f"{rule_conf_pct}%" if rule_conf_pct is not None else "N/A"

        # Format top matching rules for display
        matched_dicts = [r.to_dict() for r in matched_rules[:top_n_rules]]

        return {
            "predicted_class": predicted_class,
            "status_text": status_text,
            "status_color": status_color,
            "is_conclusive": is_conclusive,
            "probability": prob_val,
            "probability_display": prob_display,
            "probability_reason": prob_reason,
            "rule_confidence": rule_conf,
            "rule_confidence_pct": rule_conf_pct,
            "rule_confidence_display": rule_conf_display,
            "rule_confidence_label": "Rule Confidence",
            "winning_rule": winning_rule.to_dict() if winning_rule else None,
            "matched_count": len(matched_rules),
            "matched_rules": matched_dicts,
            "matching_rules": matched_dicts,
            "resolution_reason": resolution_reason,
            "conflict_detected": conflict_detected,
            "provider_tokens": sorted(list(provider_tokens)),
            "yes_rules_count": len(yes_rules),
            "no_rules_count": len(no_rules),
        }

    def predict_provider(self, feature_dict: dict, top_n_rules: int = 5) -> dict:
        """Convenience method accepting a raw feature dictionary."""
        tokens = self.discretize_features(feature_dict)
        return self.predict_tokens(tokens, top_n_rules=top_n_rules)

    def evaluate_on_test_set(self, test_df: pd.DataFrame) -> dict:
        """
        Evaluates the Apriori Decision Tree on the held-out test set with zero data leakage.
        Reports accuracy, precision, recall, F1, coverage, and inconclusive count.
        """
        if self._metrics_cache is not None:
            return self._metrics_cache

        if not self.rules:
            self.fit_from_training_data()

        total_test = len(test_df)
        predictions = []
        resolution_reasons = []
        y_true = []

        unmatched_count = 0
        tied_conflict_count = 0

        for _, row in test_df.iterrows():
            feat_dict = {col: float(row[col]) for col in FEATURE_COLUMNS}
            res = self.predict_provider(feat_dict, top_n_rules=1)
            pred = res["predicted_class"]
            predictions.append(pred)
            resolution_reasons.append(res["resolution_reason"])
            y_true.append(str(row["PotentialFraud"]).strip())

            if pred == "Inconclusive":
                if res["matched_count"] == 0:
                    unmatched_count += 1
                else:
                    tied_conflict_count += 1

        conclusive_indices = [i for i, p in enumerate(predictions) if p != "Inconclusive"]
        conclusive_count = len(conclusive_indices)
        inconclusive_count = total_test - conclusive_count
        coverage = conclusive_count / total_test if total_test > 0 else 0.0
        coverage_pct = round(coverage * 100, 2)

        if conclusive_indices:
            y_true_eval = [1 if y_true[i] == "Yes" else 0 for i in conclusive_indices]
            y_pred_eval = [1 if predictions[i] == "Yes" else 0 for i in conclusive_indices]

            acc = float(accuracy_score(y_true_eval, y_pred_eval))
            prec = float(precision_score(y_true_eval, y_pred_eval, zero_division=0))
            rec = float(recall_score(y_true_eval, y_pred_eval, zero_division=0))
            f1 = float(f1_score(y_true_eval, y_pred_eval, zero_division=0))

            cm = confusion_matrix(y_true_eval, y_pred_eval)
            tn, fp, fn, tp = cm.ravel()
        else:
            acc = prec = rec = f1 = 0.0
            tn = fp = fn = tp = 0

        metrics = {
            "model_type": "Apriori-based Rule Decision Tree",
            "is_conventional_tree": False,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "total_test_count": total_test,
            "conclusive_count": conclusive_count,
            "inconclusive_count": inconclusive_count,
            "coverage": round(coverage, 4),
            "coverage_pct": coverage_pct,
            "unmatched_count": unmatched_count,
            "tied_conflict_count": tied_conflict_count,
            "total_rules": len(self.rules),
            "fraud_rules_count": sum(1 for r in self.rules if r.target_class == "Yes"),
            "clean_rules_count": sum(1 for r in self.rules if r.target_class == "No"),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

        self._metrics_cache = metrics
        return metrics


_GLOBAL_TREE: Optional[AprioriDecisionTree] = None


def get_apriori_decision_tree(force_retrain: bool = False) -> AprioriDecisionTree:
    """Singleton getter for the trained Apriori Decision Tree."""
    global _GLOBAL_TREE
    if _GLOBAL_TREE is None or force_retrain:
        tree = AprioriDecisionTree()
        tree.fit_from_training_data(force_retrain=force_retrain)
        _GLOBAL_TREE = tree
    return _GLOBAL_TREE


def predict_with_apriori_tree(feature_dict: dict) -> dict:
    """Predict provider fraud status using the Apriori Decision Tree."""
    tree = get_apriori_decision_tree()
    return tree.predict_provider(feature_dict)


def evaluate_apriori_decision_tree(force_retrain: bool = False) -> dict:
    """Evaluates the Apriori Decision Tree on the held-out test partition."""
    tree = get_apriori_decision_tree(force_retrain=force_retrain)
    _, _, _, _, _, prov_test, full_df = get_stratified_train_test_split(
        test_size=0.2, random_state=tree.random_state
    )
    test_df = full_df[full_df["Provider"].isin(prov_test)].copy()
    return tree.evaluate_on_test_set(test_df)
