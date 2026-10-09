import os
import io
from locust import HttpUser, task, between
from PIL import Image

class LungCancerAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        # Prepare valid test image fixture bytes
        img = Image.new('RGB', (299, 299), color='gray')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        self.image_bytes = buf.getvalue()

    @task(3)
    def test_liveness(self):
        self.client.get("/health/live", name="GET /health/live")

    @task(3)
    def test_readiness(self):
        self.client.get("/health/ready", name="GET /health/ready")

    @task(2)
    def test_model_status(self):
        self.client.get("/model/status", name="GET /model/status")

    @task(5)
    def test_predict_endpoint(self):
        files = {"file": ("sample_ct.png", self.image_bytes, "image/png")}
        self.client.post("/predict", files=files, name="POST /predict")

    @task(1)
    def test_invalid_payload(self):
        files = {"file": ("test.txt", b"invalid text payload", "text/plain")}
        with self.client.post("/predict", files=files, name="POST /predict (Invalid)", catch_response=True) as response:
            if response.status_code == 400:
                response.success()
            else:
                response.failure(f"Expected 400 Bad Request, got {response.status_code}")
