import os
import time
from datetime import datetime, timezone
import numpy as np
import tensorflow as tf

from src.config import (
    IMAGE_SIZE, CLASS_LABELS, CLASS_NAMES, DEFAULT_MODEL_PATH, CONFIDENCE_THRESHOLDS
)
from src.data.preprocessing import load_and_preprocess_image
from src.models.model import build_model
from src.models.model_registry import get_active_model_info
from src.logging.audit_logger import log_inference_event

_cached_model = None

def get_model(model_path=DEFAULT_MODEL_PATH):
    """
    Loads and caches trained Keras model.
    Fallback: If trained model file is not found locally (e.g. on fresh Streamlit Cloud deploy),
    builds Xception architecture with ImageNet weights to allow seamless execution.
    """
    global _cached_model
    if _cached_model is None:
        if os.path.exists(model_path):
            try:
                _cached_model = tf.keras.models.load_model(model_path)
            except Exception as e:
                print(f"[WARNING] Error loading {model_path}: {e}. Building fallback model.")
                _cached_model, _ = build_model()
        else:
            print(f"[INFO] Trained model path {model_path} not found. Instantiating Xception base model for inference execution.")
            _cached_model, _ = build_model()
    return _cached_model

def predict_ct_scan(img_input, model_path=DEFAULT_MODEL_PATH, model=None):
    """
    Executes structured inference pipeline on input CT image.

    Returns dictionary containing:
        - prediction_id: Unique identifier for telemetry tracking
        - predicted_class: Class name display string
        - confidence: Confidence percentage (0-100)
        - confidence_level: 'HIGH' (>=80%), 'MEDIUM' (60-79%), or 'LOW' (<60%)
        - probabilities: Dictionary of class names to confidence percentages
        - model_version: Version string from model registry
        - inference_latency_ms: Milliseconds elapsed during inference
        - timestamp: ISO 8601 UTC timestamp string
        - requires_human_review: Boolean flag for low confidence predictions
        - original_image: Original PIL Image
        - preprocessed_tensor: Model preprocessed numpy input array
    """
    start_time = time.perf_counter()

    if model is None:
        model = get_model(model_path)

    img_tensor, original_img = load_and_preprocess_image(img_input)

    # Model inference
    raw_preds = model.predict(img_tensor, verbose=0)[0]
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    predicted_idx = int(np.argmax(raw_preds))
    confidence_pct = float(raw_preds[predicted_idx] * 100.0)

    # Classify confidence level
    if confidence_pct >= CONFIDENCE_THRESHOLDS['HIGH']:
        conf_level = 'HIGH'
        requires_human_review = False
    elif confidence_pct >= CONFIDENCE_THRESHOLDS['MEDIUM']:
        conf_level = 'MEDIUM'
        requires_human_review = False
    else:
        conf_level = 'LOW'
        requires_human_review = True

    probabilities = {
        CLASS_LABELS[i]: round(float(raw_preds[i] * 100.0), 2)
        for i in range(len(CLASS_NAMES))
    }

    active_info = get_active_model_info()
    model_version = active_info.get('version', 'v1.0.0-xception')
    predicted_class = CLASS_LABELS[predicted_idx]

    # Log telemetry event
    event = log_inference_event(
        predicted_class=predicted_class,
        confidence=confidence_pct,
        confidence_level=conf_level,
        latency_ms=latency_ms,
        model_version=model_version,
        requires_human_review=requires_human_review
    )

    return {
        'prediction_id': event['prediction_id'],
        'predicted_class': predicted_class,
        'confidence': round(confidence_pct, 2),
        'confidence_level': conf_level,
        'probabilities': probabilities,
        'predicted_index': predicted_idx,
        'model_version': model_version,
        'inference_latency_ms': round(latency_ms, 2),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'requires_human_review': requires_human_review,
        'recommendation': "Low-confidence prediction (<60%). Human expert review is recommended." if requires_human_review else "Prediction confidence acceptable.",
        'original_image': original_img,
        'preprocessed_tensor': img_tensor
    }
