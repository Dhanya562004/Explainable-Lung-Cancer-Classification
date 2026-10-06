import io
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel, Field
from PIL import Image

from src.config import MEDICAL_DISCLAIMER
from src.models.inference import predict_ct_scan
from src.models.model_registry import get_active_model_info
from src.evaluation.evaluate import evaluate_model
from src.monitoring.latency import get_latency_stats
from src.monitoring.performance import log_ground_truth_feedback, get_verified_performance_metrics
from src.monitoring.health import check_system_health

app = FastAPI(
    title="Explainable Lung Cancer Classification REST API",
    description="Production-oriented REST API for deep transfer learning CT scan prediction, model metadata, latency, and feedback monitoring.",
    version="1.0.0"
)

class FeedbackRequest(BaseModel):
    prediction_id: str = Field(..., json_schema_extra={"example": "pred-a1b2c3d4e5"})
    verified_class: str = Field(..., json_schema_extra={"example": "Adenocarcinoma"})
    predicted_class: str = Field(None, json_schema_extra={"example": "Adenocarcinoma"})

@app.get("/", summary="API Root")
def root():
    return {
        "title": "Explainable Lung Cancer Classification API",
        "version": "1.0.0",
        "disclaimer": MEDICAL_DISCLAIMER,
        "docs": "/docs"
    }

@app.get("/health", summary="System & Model Health Check")
def health_check():
    """
    Checks model weights availability, dataset access, disk/log readiness, and CPU/memory usage.
    """
    health = check_system_health()
    return health

@app.get("/model/info", summary="Active Model Registry Metadata")
def model_info():
    """
    Returns active production model architecture, version, parameters, and evaluation metrics.
    """
    info = get_active_model_info()
    return info

@app.post("/predict", summary="Classify Lung CT Scan Image")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accepts a chest CT scan image (PNG, JPG, JPEG) and returns structured classification probabilities,
    confidence score, latency, confidence level, and human review recommendation.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Upload a valid PNG, JPG, or JPEG image."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image file. File may be corrupted."
        )

    try:
        res = predict_ct_scan(image)
        return {
            "prediction_id": res["prediction_id"],
            "predicted_class": res["predicted_class"],
            "confidence": res["confidence"],
            "confidence_level": res["confidence_level"],
            "probabilities": res["probabilities"],
            "model_version": res["model_version"],
            "inference_latency_ms": res["inference_latency_ms"],
            "timestamp": res["timestamp"],
            "requires_human_review": res["requires_human_review"],
            "recommendation": res["recommendation"],
            "disclaimer": MEDICAL_DISCLAIMER
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference execution error occurred."
        )

@app.get("/metrics", summary="Model Evaluation Metrics")
def metrics_endpoint():
    """
    Returns official test dataset evaluation metrics (accuracy, macro F1, confusion matrix).
    """
    metrics = evaluate_model()
    return metrics

@app.get("/monitoring/summary", summary="Inference Telemetry & Latency Summary")
def monitoring_summary():
    """
    Returns total inference counts, average latency, p95 latency, error rates, and verified ground truth metrics.
    """
    lat_stats = get_latency_stats()
    feedback_stats = get_verified_performance_metrics()
    return {
        "telemetry": lat_stats,
        "verified_performance": feedback_stats
    }

@app.post("/feedback", summary="Submit Verified Ground-Truth Feedback")
def submit_feedback(req: FeedbackRequest):
    """
    Records verified ground truth feedback for a previous prediction ID.
    """
    success, msg = log_ground_truth_feedback(
        prediction_id=req.prediction_id,
        verified_class=req.verified_class,
        predicted_class=req.predicted_class
    )
    if not success:
        raise HTTPException(status_code=500, detail=msg)
    return {"status": "success", "message": msg, "prediction_id": req.prediction_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
