"""
BillWise - FastAPI Application Backend
Serves REST API endpoints for dataset overview, Random Forest evaluation,
fraud prediction, and static frontend dashboard.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import apriori_tree
import association_rules
import data_processing
import model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load features and train models so the dashboard shows real metrics immediately."""
    try:
        data_processing.process_provider_features()
        model.train_and_evaluate_model()
        association_rules.mine_association_rules()
        apriori_tree.get_apriori_decision_tree()
    except Exception as exc:
        print(f"Startup warm-up failed: {exc}")
    yield


app = FastAPI(
    title="BillWise - Healthcare Provider Fraud Detection API",
    description="Machine Learning for Medicare billing fraud detection",
    version="1.0.0",
    lifespan=lifespan,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static directory for frontend assets
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TrainRequest(BaseModel):
    n_estimators: int = Field(100, ge=10, le=500, description="Number of decision trees")
    test_size: float = Field(0.2, ge=0.1, le=0.4, description="Test split fraction")


class PredictionRequest(BaseModel):
    total_claim_count: float = Field(..., ge=0, description="Total claim count")
    total_reimb_amt: float = Field(..., ge=0, description="Total reimbursement ($)")
    mean_reimb_per_claim: float = Field(..., ge=0, description="Mean reimbursement per claim ($)")
    max_reimb_per_claim: float = Field(..., ge=0, description="Max reimbursement per claim ($)")
    inpatient_claim_count: float = Field(..., ge=0, description="Inpatient claims count")
    outpatient_claim_count: float = Field(..., ge=0, description="Outpatient claims count")
    unique_beneficiaries: float = Field(..., ge=0, description="Unique beneficiaries count")
    unique_physicians: float = Field(..., ge=0, description="Unique physicians count")
    avg_inpatient_los: float = Field(..., ge=0, description="Average inpatient length of stay (days)")


@app.get("/")
def serve_index():
    """Serves the primary web dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "BillWise API is running. static/index.html not yet found."}


@app.get("/api/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "project": "BillWise",
        "description": "Healthcare Provider Billing Fraud Detection System"
    }


@app.get("/api/overview")
def get_overview():
    """Returns dataset summary metrics and column metadata."""
    try:
        overview = data_processing.get_dataset_overview()
        return {"success": True, "data": overview}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/evaluate")
def get_model_evaluation():
    """Returns evaluation metrics on held-out test providers."""
    try:
        metrics = model.get_evaluation_metrics()
        return {"success": True, "data": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/model/train")
def retrain_model(req: TrainRequest):
    """Allows retraining the Random Forest with custom hyper-parameters."""
    try:
        metrics = model.train_and_evaluate_model(
            n_estimators=req.n_estimators,
            test_size=req.test_size
        )
        return {"success": True, "message": "Model retrained successfully", "data": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
def predict_fraud(req: PredictionRequest):
    """Predicts fraud risk and probability for given provider billing metrics."""
    try:
        feat_dict = req.model_dump()
        result = model.predict_provider_fraud(feat_dict)
        try:
            result["apriori"] = association_rules.match_rules_for_features(feat_dict)
        except Exception as apriori_exc:
            result["apriori"] = {
                "error": str(apriori_exc),
                "matched_count": 0,
                "total_rules": 0,
                "matched_rules": [],
                "top_rules": [],
            }
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sample-providers")
def get_sample_providers():
    """Returns sample held-out test providers with actual vs predicted labels."""
    try:
        samples = model.get_sample_test_providers()
        return {"success": True, "data": samples}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/provider/{provider_id}")
def get_provider_by_id(provider_id: str):
    """Returns engineered features for a provider (training set or lookup by ID)."""
    try:
        record = data_processing.get_provider_features(provider_id)
        if record is None:
            raise HTTPException(
                status_code=404,
                detail=f"Provider '{provider_id}' was not found in the training claims data.",
            )
        prediction = model.predict_provider_fraud(record["features"])
        try:
            prediction["apriori"] = association_rules.match_rules_for_features(record["features"])
        except Exception as apriori_exc:
            prediction["apriori"] = {
                "error": str(apriori_exc),
                "matched_count": 0,
                "total_rules": 0,
                "matched_rules": [],
                "top_rules": [],
            }
        return {"success": True, "data": {**record, "prediction": prediction}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/test-providers")
def list_test_provider_ids(limit: int = 50):
    """Returns provider IDs from the held-out test list CSV (labels not included in file)."""
    try:
        ids = data_processing.load_test_provider_ids(limit=limit)
        return {"success": True, "data": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/association-rules")
def get_association_rules(
    min_support: float = Query(0.03, ge=0.01, le=0.5, description="Minimum support"),
    min_confidence: float = Query(0.20, ge=0.05, le=1.0, description="Minimum confidence"),
):
    """Apriori association rules (training partition) predicting PotentialFraud=Yes."""
    try:
        rules_data = association_rules.mine_association_rules(
            min_support=min_support,
            min_confidence=min_confidence,
        )
        if rules_data.get("error"):
            raise HTTPException(status_code=500, detail=rules_data["error"])
        return {"success": True, "data": rules_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
