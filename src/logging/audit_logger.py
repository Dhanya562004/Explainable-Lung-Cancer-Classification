import os
import json
import time
import uuid
from datetime import datetime, timezone
from src.config import AUDIT_LOG_PATH, INFERENCE_LOG_PATH

def log_audit_event(event_type, details, log_path=AUDIT_LOG_PATH):
    """
    Logs structured system or audit event in JSON Lines format.
    Ensures no private patient image data is written.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    payload = {
        'request_id': str(uuid.uuid4()),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'event_type': event_type,
        'details': details
    }
    try:
        with open(log_path, 'a') as f:
            f.write(json.dumps(payload) + '\n')
    except Exception as e:
        print(f"[AUDIT LOG ERROR] Failed to write event log: {e}")
    return payload

def log_inference_event(predicted_class, confidence, confidence_level, latency_ms, model_version, requires_human_review=False):
    """
    Logs prediction metadata to monitoring inference log for telemetry,
    latency tracking, and drift detection.
    """
    os.makedirs(os.path.dirname(INFERENCE_LOG_PATH), exist_ok=True)
    payload = {
        'prediction_id': f"pred-{uuid.uuid4().hex[:10]}",
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model_version': model_version,
        'predicted_class': predicted_class,
        'confidence': round(confidence, 2),
        'confidence_level': confidence_level,
        'inference_latency_ms': round(latency_ms, 2),
        'requires_human_review': requires_human_review
    }
    try:
        with open(INFERENCE_LOG_PATH, 'a') as f:
            f.write(json.dumps(payload) + '\n')
    except Exception as e:
        print(f"[INFERENCE LOG ERROR] Failed to record inference event: {e}")
    return payload

def get_recent_audit_logs(log_path=INFERENCE_LOG_PATH, limit=50):
    """
    Reads recent logged inference events.
    """
    if not os.path.exists(log_path):
        return []
    logs = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))
        return logs[-limit:]
    except Exception:
        return []
