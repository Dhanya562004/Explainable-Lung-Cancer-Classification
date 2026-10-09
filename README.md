# 🫁 Explainable Lung Cancer Classification & Production MLOps Engine

[![CI Quality & Docker Verification](https://github.com/Dhanya562004/Explainable-Lung-Cancer-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhanya562004/Explainable-Lung-Cancer-Classification/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Deep Learning](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> ⚠️ **EDUCATIONAL & RESEARCH DISCLAIMER**  
> This software repository is designed strictly for **educational, demonstration, and software engineering portfolio purposes**. It is **not** a clinically validated medical diagnostic device and is **not** intended for clinical diagnosis, patient management, or medical decision-making.

---

## 📐 System Architecture Overview

```
                          ┌─────────────────────────┐
                          │   Streamlit Web UI      │
                          │   (Port 8501 / NodePort)│
                          └────────────┬────────────┘
                                       │ HTTP / REST
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FastAPI Backend Server                           │
│                            (Port 8000 / ClusterIP)                          │
├───────────────────┬───────────────────────────┬─────────────────────────────┤
│   Health Probes   │    Inference Engine       │     Telemetry & Monitoring  │
│  - /health/live   │  - Xception Backbone      │   - Prometheus Exporter    │
│  - /health/ready  │  - Grad-CAM Interpretability- Audit JSON Logs       │
│  - /model/status  │  - Confidence Scoring     │   - Latency & Drift Tracker │
└───────────────────┴───────────────────────────┴─────────────────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   Model Registry & Artifacts      │
                     │  - models/best_model.keras        │
                     │  - models/model_registry.json     │
                     └───────────────────────────────────┘
```

---

## 🔑 Key Features & Technical Highlights

- **Deep Transfer Learning:** 4-class chest CT scan classification (*Adenocarcinoma, Large Cell Carcinoma, Normal, Squamous Cell Carcinoma*) fine-tuned on the Xception backbone.
- **Grad-CAM Visual Explainability:** Computes spatial gradient activation heatmaps overlaid onto raw CT scans to highlight anatomical regions influencing predictions.
- **Production REST API:** Built with FastAPI, featuring strict input validation, image MIME type checking, upload payload limits (10MB boundary), and standardized HTTP error responses.
- **Enterprise Observability:** Thread-safe Prometheus metrics endpoint (`/metrics/prometheus`), structured JSON application logs (`api_server`), real-time latency percentiles, and input data drift detection.
- **Microservice Containerization:** Non-root security context (`appuser`, UID `10001`), multi-stage Docker builds, liveness/readiness container health checks, and Docker Compose orchestration.
- **Kubernetes & Cloud Readiness:** Production manifests in `deploy/k8s/` (ConfigMap, Deployment, Service, Kustomize) with resource limits and rolling update strategies, alongside Azure Container Apps (ACA) IaC Bicep templates in `deploy/azure/`.

---

## 🤖 Deep Learning & Grad-CAM Interpretability

| Property | Specification |
| :--- | :--- |
| **Model Architecture** | Xception Deep Transfer Learning (ImageNet pre-trained base) |
| **Input Shape** | `299 x 299 x 3` RGB Image Tensor |
| **Classes** | `Adenocarcinoma`, `Large Cell Carcinoma`, `Normal`, `Squamous Cell Carcinoma` |
| **Test Accuracy** | `76.51%` (315-image isolated test set) |
| **Macro F1-Score** | `0.7859` |
| **Explainability** | Grad-CAM feature activation maps extracted from last conv layer |
| **Confidence Gating** | Low-confidence inferences (`<60%`) trigger human expert review alerts |

---

## 🚀 Quickstart: Local Setup & Running

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/Dhanya562004/Explainable-Lung-Cancer-Classification.git
cd Explainable-Lung-Cancer-Classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Local Servers
```bash
# Start FastAPI backend API server (Port 8000)
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# Start Streamlit UI interface (Port 8501)
streamlit run app.py
```

---

## 🌐 REST API Endpoint Reference

| Method | Endpoint | Description | Status Codes |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | API Root information and disclaimer | `200` |
| `GET` | `/health/live` | Process liveness probe | `200` |
| `GET` | `/health/ready` | Model artifact readiness probe | `200`, `503` |
| `GET` | `/model/status` | Diagnostic model parameters and load state | `200` |
| `GET` | `/model/info` | Active model version & registry metrics | `200` |
| `POST` | `/predict` | Classify uploaded CT image | `200`, `400`, `413`, `500` |
| `GET` | `/metrics` | Official model test dataset evaluation metrics | `200` |
| `GET` | `/metrics/prometheus` | Prometheus operational metrics scraper format | `200` |
| `GET` | `/monitoring/summary` | Telemetry latency summary & ground-truth metrics | `200` |
| `POST` | `/feedback` | Submit verified ground-truth clinical feedback | `200`, `500` |

---

## 🧪 Testing & Code Quality Verification

```bash
# Run complete Pytest test suite (23 unit & integration tests)
pytest -v --tb=short

# Run Flake8 linter
flake8 src/ tests/
```

---

## 🐳 Docker & Docker Compose Setup

```bash
# Build Docker container image locally
docker build -t explainable-lung-cancer-api:v1.0.0 .

# Run single container
docker run -p 8000:8000 -p 8501:8501 explainable-lung-cancer-api:v1.0.0

# Or launch multi-service container environment with Docker Compose
docker-compose up --build
```

---

## ☸️ Kubernetes Deployment

Manifests are located under [`deploy/k8s/`](file:///c:/Users/Deeksha/OneDrive/Desktop/Explainable-Lung-Cancer-Classification-main/deploy/k8s/):

```bash
# Apply Kubernetes manifests via Kustomize
kubectl apply -k deploy/k8s/

# Monitor rollout status
kubectl rollout status deployment/fastapi-deployment
kubectl rollout status deployment/streamlit-deployment
```
Detailed instructions are provided in [`deploy/k8s/README.md`](file:///c:/Users/Deeksha/OneDrive/Desktop/Explainable-Lung-Cancer-Classification-main/deploy/k8s/README.md).

---

## ☁️ Azure Container Apps Cloud Deployment

Infrastructure as Code templates and GitHub Action dispatch workflows are available under [`deploy/azure/`](file:///c:/Users/Deeksha/OneDrive/Desktop/Explainable-Lung-Cancer-Classification-main/deploy/azure/):

```bash
# Deploy to Azure via Azure CLI and Bicep
az group create --name rg-lungcancer-prod --location eastus
az deployment group create \
  --resource-group rg-lungcancer-prod \
  --template-file deploy/azure/azure-container-apps.bicep
```
Detailed instructions and pricing guides are provided in [`deploy/azure/AZURE_DEPLOYMENT.md`](file:///c:/Users/Deeksha/OneDrive/Desktop/Explainable-Lung-Cancer-Classification-main/deploy/azure/AZURE_DEPLOYMENT.md).

---

## 📊 Measured Load Test Performance Results

Empirical performance benchmark results measured locally:

| Performance Metric | Measured Value |
| :--- | :--- |
| **Total Test Requests** | `15` concurrent prediction requests |
| **Tested Concurrency** | `3` parallel workers |
| **Warm-up Latency (Cold Start)** | `4,637.4 ms` |
| **Throughput (RPS)** | `2.80 req/sec` |
| **Median (p50) Latency** | `975.0 ms` |
| **p95 Latency** | `1,464.1 ms` |
| **p99 Latency** | `1,513.5 ms` |
| **Error Rate** | `0.00%` |

Run load tests locally using:
```bash
python tests/performance/run_performance_test.py --url http://127.0.0.1:8000 --requests 15 --concurrency 3
```

---

## 📌 Implementation Verification Summary

| Phase & Feature Component | Verification Status |
| :--- | :--- |
| **Baseline & 23 Pytest Tests** | ✅ Verified locally (`23 passed in 108s`) |
| **FastAPI Reliability & Health Probes** | ✅ Verified (`/health/live`, `/health/ready`, `/model/status`, `/metrics/prometheus`) |
| **GitHub Actions CI (`ci.yml`)** | ✅ Configured for linting, testing, and Docker build |
| **Docker & Non-root User Container** | ✅ Verified build and execution (`appuser`, UID `10001`) |
| **Kubernetes Manifests (`deploy/k8s/`)** | 📄 Prepared & syntax-validated (`deployment`, `service`, `configmap`, `kustomize`) |
| **Locust Load Testing Suite** | ✅ Executed locally & benchmark report generated (`results/performance_report.md`) |
| **Azure Container Apps Bicep Setup** | 📄 Prepared IaC templates & OIDC workflow ready for user subscription |
