import pytest
import numpy as np
from src.monitoring.drift import compute_image_stats, detect_data_drift
from src.monitoring.latency import get_latency_stats
from src.monitoring.performance import get_verified_performance_metrics, log_ground_truth_feedback

def test_compute_image_stats_and_drift():
    tensor = np.zeros((1, 299, 299, 3), dtype=np.float32)
    stats = compute_image_stats(tensor)
    assert 'mean' in stats
    assert 'std' in stats

    drift = detect_data_drift(stats)
    assert 'drift_status' in drift
    assert drift['drift_status'] in ['LOW DRIFT', 'MEDIUM DRIFT', 'HIGH DRIFT']
    assert 'drift_score' in drift
    assert 'disclaimer' in drift

def test_latency_stats():
    stats = get_latency_stats(log_path="non_existent_log.jsonl")
    assert stats['total_predictions'] == 0
    assert stats['avg_latency_ms'] == 0.0

def test_feedback_logging_and_performance():
    success, msg = log_ground_truth_feedback("test-pred-123", "Adenocarcinoma", "Adenocarcinoma")
    assert success is True
    
    verified_perf = get_verified_performance_metrics()
    assert 'has_verified_data' in verified_perf
    assert 'notice' in verified_perf
