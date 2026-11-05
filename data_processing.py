"""
BillWise - Data Processing & Feature Engineering Module
Handles loading raw Medicare claims CSV files, cleaning, joining,
and aggregating claims to provider-level feature vectors.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_PROV_PATH = os.path.join(DATA_DIR, "Train-1542865627584.csv")
TRAIN_INP_PATH = os.path.join(DATA_DIR, "Train_Inpatientdata-1542865627584.csv")
TRAIN_OUTP_PATH = os.path.join(DATA_DIR, "Train_Outpatientdata-1542865627584.csv")
TEST_PROV_PATH = os.path.join(DATA_DIR, "Test-1542969243754.csv")
CACHE_PATH = os.path.join(DATA_DIR, "processed_providers_cache.csv")

FEATURE_COLUMNS = [
    "total_claim_count",
    "total_reimb_amt",
    "mean_reimb_per_claim",
    "max_reimb_per_claim",
    "inpatient_claim_count",
    "outpatient_claim_count",
    "unique_beneficiaries",
    "unique_physicians",
    "avg_inpatient_los"
]

FEATURE_LABELS = {
    "total_claim_count": "Total Claim Count",
    "total_reimb_amt": "Total Reimbursement Amount ($)",
    "mean_reimb_per_claim": "Average Reimbursement per Claim ($)",
    "max_reimb_per_claim": "Maximum Reimbursement per Claim ($)",
    "inpatient_claim_count": "Inpatient Claim Count",
    "outpatient_claim_count": "Outpatient Claim Count",
    "unique_beneficiaries": "Unique Beneficiaries Count",
    "unique_physicians": "Unique Physicians Involved",
    "avg_inpatient_los": "Avg. Inpatient Length of Stay (Days)"
}


def load_raw_data():
    """Load raw CSV files from data directory."""
    if not os.path.exists(TRAIN_PROV_PATH):
        raise FileNotFoundError(f"Missing file: {TRAIN_PROV_PATH}")
    if not os.path.exists(TRAIN_INP_PATH):
        raise FileNotFoundError(f"Missing file: {TRAIN_INP_PATH}")
    if not os.path.exists(TRAIN_OUTP_PATH):
        raise FileNotFoundError(f"Missing file: {TRAIN_OUTP_PATH}")

    train_prov = pd.read_csv(TRAIN_PROV_PATH)
    inp = pd.read_csv(TRAIN_INP_PATH)
    outp = pd.read_csv(TRAIN_OUTP_PATH)
    return train_prov, inp, outp


def get_dataset_overview():
    """Returns dataset summary statistics for the frontend overview section."""
    train_prov, inp, outp = load_raw_data()
    total_providers = len(train_prov)
    fraud_counts = train_prov["PotentialFraud"].value_counts().to_dict()
    fraud_yes = int(fraud_counts.get("Yes", 0))
    fraud_no = int(fraud_counts.get("No", 0))
    fraud_pct = round((fraud_yes / total_providers) * 100, 2) if total_providers > 0 else 0.0

    return {
        "total_providers": total_providers,
        "fraud_yes": fraud_yes,
        "fraud_no": fraud_no,
        "fraud_percentage": fraud_pct,
        "inpatient_claims": len(inp),
        "outpatient_claims": len(outp),
        "total_claims": len(inp) + len(outp),
        "features": list(FEATURE_LABELS.items())
    }


def process_provider_features(force_recompute=False):
    """
    Aggregates inpatient and outpatient claims to provider-level rows.
    Uses cache if available for fast application startup.
    """
    if not force_recompute and os.path.exists(CACHE_PATH):
        try:
            cached_df = pd.read_csv(CACHE_PATH)
            # Verify required columns exist
            required = ["Provider", "PotentialFraud"] + FEATURE_COLUMNS
            if all(col in cached_df.columns for col in required):
                return cached_df
        except Exception as e:
            print(f"Cache load failed ({e}), recomputing...")

    train_prov, inp, outp = load_raw_data()

    # Tag claim types
    inp["Is_Inpatient"] = 1
    outp["Is_Inpatient"] = 0

    # Calculate inpatient length of stay (LOS) in days safely
    adm_dt = pd.to_datetime(inp["AdmissionDt"], errors="coerce")
    dis_dt = pd.to_datetime(inp["DischargeDt"], errors="coerce")
    inp["Inpatient_LOS"] = (dis_dt - adm_dt).dt.days.clip(lower=0)
    outp["Inpatient_LOS"] = np.nan

    # Combine claims
    claims = pd.concat([inp, outp], ignore_index=True)
    claims["InscClaimAmtReimbursed"] = pd.to_numeric(
        claims["InscClaimAmtReimbursed"], errors="coerce"
    ).fillna(0.0)

    # Provider-level aggregation
    grouped = claims.groupby("Provider")
    agg_df = grouped.agg(
        total_claim_count=("ClaimID", "count"),
        total_reimb_amt=("InscClaimAmtReimbursed", "sum"),
        mean_reimb_per_claim=("InscClaimAmtReimbursed", "mean"),
        max_reimb_per_claim=("InscClaimAmtReimbursed", "max"),
        inpatient_claim_count=("Is_Inpatient", "sum"),
        unique_beneficiaries=("BeneID", "nunique"),
        avg_inpatient_los=("Inpatient_LOS", "mean")
    ).reset_index()

    # Derived claim counts
    agg_df["outpatient_claim_count"] = agg_df["total_claim_count"] - agg_df["inpatient_claim_count"]
    agg_df["avg_inpatient_los"] = agg_df["avg_inpatient_los"].fillna(0.0).round(2)
    agg_df["mean_reimb_per_claim"] = agg_df["mean_reimb_per_claim"].round(2)
    agg_df["total_reimb_amt"] = agg_df["total_reimb_amt"].round(2)

    # Unique physicians aggregation across Attending, Operating, Other
    phys_df = pd.melt(
        claims[["Provider", "AttendingPhysician", "OperatingPhysician", "OtherPhysician"]],
        id_vars=["Provider"],
        value_name="Physician"
    ).dropna(subset=["Physician"])
    phys_df = phys_df[phys_df["Physician"].astype(str).str.strip() != ""]
    phys_counts = phys_df.groupby("Provider")["Physician"].nunique().reset_index().rename(
        columns={"Physician": "unique_physicians"}
    )

    agg_df = pd.merge(agg_df, phys_counts, on="Provider", how="left")
    agg_df["unique_physicians"] = agg_df["unique_physicians"].fillna(0).astype(int)

    # Merge with Provider ground truth labels
    final_df = pd.merge(train_prov, agg_df, on="Provider", how="left").fillna(0)

    # Ensure correct dtypes
    int_cols = ["total_claim_count", "inpatient_claim_count", "outpatient_claim_count",
                "unique_beneficiaries", "unique_physicians"]
    for col in int_cols:
        final_df[col] = final_df[col].astype(int)

    # Save cache for fast reload
    try:
        final_df.to_csv(CACHE_PATH, index=False)
    except Exception as e:
        print(f"Warning: could not write cache file: {e}")

    return final_df


def get_stratified_train_test_split(test_size=0.2, random_state=42):
    """
    Performs provider-level stratified split to ensure zero data leakage.
    Returns:
        X_train, X_test, y_train, y_test, prov_train, prov_test
    """
    df = process_provider_features()
    X = df[FEATURE_COLUMNS].copy()
    y = (df["PotentialFraud"] == "Yes").astype(int)
    providers = df["Provider"]

    X_train, X_test, y_train, y_test, prov_train, prov_test = train_test_split(
        X, y, providers, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, prov_train, prov_test, df


def load_test_provider_ids(limit=None):
    """Load provider IDs from the competition test list (no fraud labels in file)."""
    if not os.path.exists(TEST_PROV_PATH):
        return []
    test_df = pd.read_csv(TEST_PROV_PATH)
    ids = test_df["Provider"].astype(str).tolist()
    if limit is not None:
        return ids[: int(limit)]
    return ids


def get_provider_features(provider_id):
    """
    Look up a single provider's engineered feature vector and known label (if in training set).
    Returns None if the provider has no claims in the training data.
    """
    df = process_provider_features()
    provider_id = str(provider_id).strip()
    match = df[df["Provider"].astype(str) == provider_id]
    if match.empty:
        return None

    row = match.iloc[0]
    features = {col: float(row[col]) for col in FEATURE_COLUMNS}
    actual = row["PotentialFraud"] if "PotentialFraud" in row.index else None
    return {
        "Provider": provider_id,
        "PotentialFraud": actual,
        "features": features,
    }
