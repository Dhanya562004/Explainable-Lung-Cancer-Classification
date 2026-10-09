import io
import time
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from PIL import Image

from src.config import MEDICAL_DISCLAIMER, DEFAULT_MODEL_PATH
from src.models.inference import predict_ct_scan, get_model
from src.models.model_registry import get_active_model_info
from src.evaluation.evaluate import evaluate_model
from src.monitoring.latency import get_latency_stats
from src.monitoring.performance import log_ground_truth_feedback, get_verified_performance_metrics
from src.monitoring.health import check_system_health
from src.monitoring.telemetry import metrics_manager
from src.logging.structured_logger import app_logger

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit

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

@app.get("/health/live", summary="Liveness Probe")
def health_live():
    """
    Lightweight probe verifying that the API server process is alive and responding.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/health/ready", summary="Readiness Probe")
def health_ready(response: Response):
    """
    Reports whether required model artifacts and core dependencies are ready to process predictions.
    """
    try:
        model = get_model()
        is_ready = model is not None
    except Exception as e:
        app_logger.error(f"Readiness check failed: {e}")
        is_ready = False

    if is_ready:
        return {
            "status": "ready",
            "model_loaded": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "model_loaded": False,
            "reason": "Model artifact failed to initialize",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/health", summary="System & Model Health Check")
def health_check():
    """
    Checks model weights availability, dataset access, disk/log readiness, and CPU/memory usage.
    """
    health = check_system_health()
    return health

@app.get("/model/status", summary="Model Operational Diagnostic Status")
def model_status():
    """
    Returns operational diagnostic information without exposing sensitive system paths.
    """
    info = get_active_model_info()
    model_loaded = False
    try:
        model = get_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False

    return {
        "active_version": info.get("version", "v1.0.0-xception"),
        "architecture": info.get("architecture", "Xception"),
        "model_loaded": model_loaded,
        "parameters": {
            "total": info.get("num_parameters", 21092804),
            "trainable": info.get("trainable_parameters", 3613060)
        },
        "input_shape": [299, 299, 3],
        "framework": "TensorFlow / Keras"
    }

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
    start_time = time.perf_counter()

    # Validate file type / content-type header
    if file.content_type and not file.content_type.startswith("image/"):
        metrics_manager.record_prediction(status="failure", latency_ms=0.0)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Upload a valid PNG, JPG, or JPEG image."
        )

    # Read content with upload size boundary enforcement
    try:
        contents = await file.read()
    except Exception as e:
        metrics_manager.record_prediction(status="failure", latency_ms=0.0)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read uploaded file content."
        )

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        metrics_manager.record_prediction(status="failure", latency_ms=0.0)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File payload exceeds maximum limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB."
        )

    if len(contents) == 0:
        metrics_manager.record_prediction(status="failure", latency_ms=0.0)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file upload provided."
        )

    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        metrics_manager.record_prediction(status="failure", latency_ms=0.0)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image file. File may be corrupted or unreadable."
        )

    try:
        res = predict_ct_scan(image)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        metrics_manager.record_prediction(
            status="success",
            latency_ms=latency_ms,
            confidence_level=res.get("confidence_level", "MEDIUM")
        )

        app_logger.info(
            f"Prediction completed: id={res['prediction_id']} class={res['predicted_class']} conf={res['confidence']:.2f}%"
        )

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
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        metrics_manager.record_prediction(status="failure", latency_ms=latency_ms)
        app_logger.error(f"Inference execution failed: {e}")
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

@app.get("/metrics/prometheus", response_class=PlainTextResponse, summary="Prometheus Operational Metrics")
def prometheus_metrics_endpoint():
    """
    Exposes operational metrics in standard Prometheus text format for scrapers.
    """
    return metrics_manager.generate_prometheus_format()

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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)
    return {"status": "success", "message": msg, "prediction_id": req.prediction_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
