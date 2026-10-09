import os
import pytest
from src.models.inference import get_model
from src.monitoring.health import check_system_health

def test_model_loading_fallback():
    # Attempting to load model from non-existent path triggers fallback
    model = get_model(model_path="non_existent_model.keras")
    assert model is not None
    # Model should accept standard input shape (1, 299, 299, 3)
    assert model.input_shape == (None, 299, 299, 3)

def test_system_health_check_structure():
    health = check_system_health()
    assert "status" in health
    assert health["status"] in ["HEALTHY", "DEGRADED"]
    assert "checks" in health
    checks = health["checks"]
    assert "model_status" in checks
    assert "dataset_accessible" in checks
