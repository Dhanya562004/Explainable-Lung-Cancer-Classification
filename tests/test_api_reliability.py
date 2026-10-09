import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from src.api.server import app

client = TestClient(app)

def test_liveness_probe():
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"
    assert "timestamp" in data

def test_readiness_probe():
    response = client.get("/health/ready")
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_model_status():
    response = client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert "active_version" in data
    assert "architecture" in data
    assert "parameters" in data
    assert "framework" in data

def test_prometheus_metrics():
    response = client.get("/metrics/prometheus")
    assert response.status_code == 200
    content = response.text
    assert "api_predictions_total" in content
    assert "api_prediction_latency_seconds_total" in content

def test_predict_invalid_content_type():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"this is text not image", "text/plain")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data

def test_predict_corrupt_image():
    response = client.post(
        "/predict",
        files={"file": ("corrupt.png", b"not a valid png binary content", "image/png")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "Could not decode image file" in data["detail"]

def test_predict_empty_file():
    response = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "Empty file upload" in data["detail"]

def test_predict_payload_too_large():
    # 11 MB fake payload
    oversized_data = b"0" * (11 * 1024 * 1024)
    response = client.post(
        "/predict",
        files={"file": ("large.png", oversized_data, "image/png")}
    )
    assert response.status_code == 413
    data = response.json()
    assert "exceeds maximum limit" in data["detail"]

def test_valid_prediction_and_prometheus_counter():
    img = Image.new('RGB', (299, 299), color='blue')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    response = client.post(
        "/predict",
        files={"file": ("test_valid.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]

    # Verify Prometheus counters incremented
    metrics_res = client.get("/metrics/prometheus")
    assert metrics_res.status_code == 200
    assert 'api_predictions_total{status="success"}' in metrics_res.text
