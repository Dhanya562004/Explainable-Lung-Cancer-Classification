import io
from fastapi.testclient import TestClient
from PIL import Image
from src.api.server import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "disclaimer" in data

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data

def test_api_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "architecture" in data

def test_api_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "test_accuracy" in data

def test_api_predict():
    img = Image.new('RGB', (299, 299), color='green')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    response = client.post(
        "/predict",
        files={"file": ("test.png", img_bytes, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "model_version" in data
    assert "inference_latency_ms" in data

def test_api_feedback():
    payload = {
        "prediction_id": "test-pred-api-1",
        "verified_class": "Normal",
        "predicted_class": "Normal"
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
