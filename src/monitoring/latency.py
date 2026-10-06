import os
import json
import numpy as np
from src.config import INFERENCE_LOG_PATH
from src.logging.audit_logger import get_recent_audit_logs

def get_latency_stats(log_path=INFERENCE_LOG_PATH):
    """
    Calculates operational latency and telemetry statistics from inference logs.

    Returns:
        dict: total_predictions, avg_latency_ms, p95_latency_ms, min_latency_ms, max_latency_ms,
              error_rate, low_confidence_rate, class_distribution
    """
    logs = get_recent_audit_logs(log_path=log_path, limit=1000)
    
    if not logs:
        return {
            'total_predictions': 0,
            'avg_latency_ms': 0.0,
            'p95_latency_ms': 0.0,
            'min_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'low_confidence_count': 0,
            'low_confidence_rate_pct': 0.0,
            'class_distribution': {}
        }

    latencies = [l.get('inference_latency_ms', 0.0) for l in logs if 'inference_latency_ms' in l]
    classes = [l.get('predicted_class', 'Unknown') for l in logs if 'predicted_class' in l]
    low_conf_count = sum(1 for l in logs if l.get('requires_human_review', False))

    class_dist = {}
    for c in classes:
        class_dist[c] = class_dist.get(c, 0) + 1

    total = len(logs)
    avg_lat = float(np.mean(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
    min_lat = float(np.min(latencies)) if latencies else 0.0
    max_lat = float(np.max(latencies)) if latencies else 0.0

    return {
        'total_predictions': total,
        'avg_latency_ms': round(avg_lat, 2),
        'p95_latency_ms': round(p95_lat, 2),
        'min_latency_ms': round(min_lat, 2),
        'max_latency_ms': round(max_lat, 2),
        'low_confidence_count': low_conf_count,
        'low_confidence_rate_pct': round((low_conf_count / total) * 100, 2) if total > 0 else 0.0,
        'class_distribution': class_dist
    }
