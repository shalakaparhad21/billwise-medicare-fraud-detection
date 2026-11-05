"""
BillWise - Apriori Association Rule Mining (backend)
Mines rules from discretized provider-level features on the training partition only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from data_processing import (
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    get_stratified_train_test_split,
)

FRAUD_CONSEQUENT = "PotentialFraud=Yes"
BIN_LABELS = ("Low", "Medium", "High")

_CACHED_RESULT = None
_BIN_EDGES = None


def _labels_for_bin_count(n_bins: int):
    if n_bins <= 1:
        return ["Medium"]
    if n_bins == 2:
        return ["Low", "High"]
    return list(BIN_LABELS[:n_bins]) if n_bins <= 3 else list(BIN_LABELS)


def _compute_bin_edges(provider_df: pd.DataFrame) -> dict:
    """Quantile edges from training providers, reused to discretize new inputs."""
    edges = {}
    for col in FEATURE_COLUMNS:
        numeric = pd.to_numeric(provider_df[col], errors="coerce").fillna(0.0)
        try:
            _, bins = pd.qcut(numeric, q=3, retbins=True, duplicates="drop")
        except ValueError:
            _, bins = pd.cut(numeric, bins=3, retbins=True, duplicates="drop")
        bins = np.unique(np.asarray(bins, dtype=float))
        if len(bins) < 2:
            lo, hi = float(numeric.min()), float(numeric.max())
            if hi <= lo:
                hi = lo + 1.0
            bins = np.array([lo, hi], dtype=float)
        edges[col] = bins
    return edges


def _discretize_series(series: pd.Series, feature_name: str, bins=None) -> pd.Series:
    """Map numeric feature values to Low / Medium / High bins."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if bins is None:
        try:
            binned = pd.qcut(numeric, q=3, labels=BIN_LABELS, duplicates="drop")
        except ValueError:
            binned = pd.cut(numeric, bins=3, labels=BIN_LABELS, duplicates="drop")
    else:
        n_bins = max(len(bins) - 1, 1)
        labels = _labels_for_bin_count(n_bins)
        clipped = numeric.clip(lower=float(bins[0]), upper=float(bins[-1]))
        binned = pd.cut(clipped, bins=bins, labels=labels, include_lowest=True)

    if binned.isna().all():
        binned = pd.Series(["Medium"] * len(numeric), index=numeric.index)

    binned = binned.astype(str).replace("nan", "Medium")
    return binned.apply(lambda level: f"{feature_name}_{level}")


def _build_one_hot_frame(provider_df: pd.DataFrame, bin_edges: dict | None = None) -> pd.DataFrame:
    """One-hot encode discretized features and fraud label for mlxtend Apriori."""
    items = {}

    for col in FEATURE_COLUMNS:
        bins = None if bin_edges is None else bin_edges.get(col)
        item_ids = _discretize_series(provider_df[col], col, bins=bins)
        for level in BIN_LABELS:
            token = f"{col}_{level}"
            items[token] = item_ids == token

    items[FRAUD_CONSEQUENT] = provider_df["PotentialFraud"].astype(str).str.strip() == "Yes"
    items["PotentialFraud=No"] = provider_df["PotentialFraud"].astype(str).str.strip() == "No"

    frame = pd.DataFrame(items)
    return frame.astype(bool)


def _format_item(token: str) -> str:
    """Human-readable antecedent/consequent label."""
    if token.startswith("PotentialFraud"):
        return token.replace("=", " = ")
    if token.endswith("_Low"):
        base = token[: -len("_Low")]
        return f"{FEATURE_LABELS.get(base, base)} (Low)"
    if token.endswith("_Medium"):
        base = token[: -len("_Medium")]
        return f"{FEATURE_LABELS.get(base, base)} (Medium)"
    if token.endswith("_High"):
        base = token[: -len("_High")]
        return f"{FEATURE_LABELS.get(base, base)} (High)"
    return token


def _interpret_rule(antecedents, confidence_pct: float, lift: float) -> str:
    parts = [_format_item(a) for a in antecedents]
    pattern = " AND ".join(parts) if parts else "(empty pattern)"
    return (
        f"When {pattern}, estimated fraud likelihood is {confidence_pct:.1f}% "
        f"(lift {lift:.2f}x vs baseline)."
    )


def mine_association_rules(
    min_support: float = 0.03,
    min_confidence: float = 0.20,
    max_len: int = 4,
    top_n: int = 50,
    random_state: int = 42,
):
    """
    Run Apriori on training providers only; return rules whose consequent includes
    PotentialFraud=Yes.
    """
    global _BIN_EDGES, _CACHED_RESULT
    try:
        _, _, _, _, prov_train, _, full_df = get_stratified_train_test_split(
            test_size=0.2, random_state=random_state
        )
        train_df = full_df[full_df["Provider"].isin(prov_train)].copy()
        if train_df.empty:
            return {
                "error": "Training partition is empty; cannot mine association rules.",
                "total_rules": 0,
                "base_fraud_rate": 0.0,
                "min_support": min_support,
                "min_confidence": min_confidence,
                "train_provider_count": 0,
                "rules": [],
            }

        base_fraud_rate = float((train_df["PotentialFraud"] == "Yes").mean())
        _BIN_EDGES = _compute_bin_edges(train_df)
        one_hot = _build_one_hot_frame(train_df, bin_edges=_BIN_EDGES)

        frequent = apriori(
            one_hot,
            min_support=min_support,
            use_colnames=True,
            max_len=max_len,
            low_memory=True,
        )
        if frequent.empty:
            return {
                "total_rules": 0,
                "base_fraud_rate": round(base_fraud_rate, 4),
                "min_support": min_support,
                "min_confidence": min_confidence,
                "train_provider_count": int(len(train_df)),
                "rules": [],
                "message": "No frequent itemsets at the given minimum support.",
            }

        try:
            rules_df = association_rules(
                frequent,
                metric="confidence",
                min_threshold=min_confidence,
                num_itemsets=len(frequent),
            )
        except TypeError:
            rules_df = association_rules(
                frequent,
                metric="confidence",
                min_threshold=min_confidence,
            )
        if rules_df.empty:
            return {
                "total_rules": 0,
                "base_fraud_rate": round(base_fraud_rate, 4),
                "min_support": min_support,
                "min_confidence": min_confidence,
                "train_provider_count": int(len(train_df)),
                "rules": [],
                "message": "No association rules met the minimum confidence threshold.",
            }

        def predicts_fraud_only(consequents) -> bool:
            return set(consequents) == {FRAUD_CONSEQUENT}

        rules_df = rules_df[rules_df["consequents"].apply(predicts_fraud_only)].copy()
        rules_df = rules_df.sort_values(["lift", "confidence", "support"], ascending=False)

        output_rules = []
        for _, row in rules_df.head(top_n).iterrows():
            antecedents = sorted([a for a in row["antecedents"] if a != FRAUD_CONSEQUENT])
            consequents = sorted(list(row["consequents"]))
            support = float(row["support"])
            confidence = float(row["confidence"])
            lift = float(row["lift"])
            confidence_pct = round(confidence * 100, 2)
            support_pct = round(support * 100, 2)
            lift_rounded = round(lift, 3)

            output_rules.append(
                {
                    "antecedents": [_format_item(a) for a in antecedents],
                    "antecedent_tokens": antecedents,
                    "consequents": [_format_item(c) for c in consequents],
                    "consequent_tokens": consequents,
                    "support": round(support, 4),
                    "support_pct": support_pct,
                    "confidence": round(confidence, 4),
                    "confidence_pct": confidence_pct,
                    "lift": lift_rounded,
                    "interpretation": _interpret_rule(antecedents, confidence_pct, lift_rounded),
                }
            )

        result = {
            "total_rules": len(output_rules),
            "base_fraud_rate": round(base_fraud_rate, 4),
            "min_support": min_support,
            "min_confidence": min_confidence,
            "train_provider_count": int(len(train_df)),
            "partition": "training_only",
            "rules": output_rules,
        }
        _CACHED_RESULT = result
        return result

    except ImportError as exc:
        return {
            "error": f"Missing dependency: {exc}. Install mlxtend.",
            "total_rules": 0,
            "base_fraud_rate": 0.0,
            "rules": [],
        }
    except Exception as exc:
        return {
            "error": f"Association rule mining failed: {exc}",
            "total_rules": 0,
            "base_fraud_rate": 0.0,
            "rules": [],
        }


def _provider_item_tokens(feature_dict: dict) -> set:
    """Discretize one provider into Apriori item tokens using training bin edges."""
    global _BIN_EDGES
    if _BIN_EDGES is None:
        mine_association_rules()
    tokens = set()
    for col in FEATURE_COLUMNS:
        try:
            value = float(feature_dict.get(col, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        bins = None if _BIN_EDGES is None else _BIN_EDGES.get(col)
        token_series = _discretize_series(pd.Series([value]), col, bins=bins)
        tokens.add(str(token_series.iloc[0]))
    return tokens


def match_rules_for_features(feature_dict: dict, top_n: int = 5) -> dict:
    """
    Return Apriori fraud rules whose antecedents match this provider's
    Low/Medium/High billing bins. Always includes mined-rule metadata so
    the prediction panel can show association output even if none fire.
    """
    global _CACHED_RESULT
    if _CACHED_RESULT is None:
        mined = mine_association_rules()
    else:
        mined = _CACHED_RESULT

    if mined.get("error"):
        return {
            "error": mined["error"],
            "matched_count": 0,
            "total_rules": 0,
            "base_fraud_rate": 0.0,
            "matched_rules": [],
            "top_rules": [],
            "itemset": [],
        }

    tokens = _provider_item_tokens(feature_dict)
    matched = []
    for rule in mined.get("rules", []):
        antecedents = set(rule.get("antecedent_tokens") or [])
        if antecedents and antecedents.issubset(tokens):
            matched.append(rule)

    return {
        "matched_count": len(matched),
        "total_rules": mined.get("total_rules", 0),
        "base_fraud_rate": mined.get("base_fraud_rate", 0.0),
        "train_provider_count": mined.get("train_provider_count", 0),
        "itemset": sorted(_format_item(t) for t in tokens),
        "matched_rules": matched[:top_n],
        "top_rules": (mined.get("rules") or [])[:top_n],
        "message": mined.get("message"),
    }
