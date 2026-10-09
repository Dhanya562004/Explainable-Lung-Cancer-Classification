# ☸️ Kubernetes Deployment Guide

This directory contains production-ready Kubernetes manifests for deploying the **Explainable Lung Cancer Classification** microservices architecture on local (Minikube / Kind) or managed cloud Kubernetes clusters (AKS / EKS / GKE).

---

## 📁 Manifest Directory Structure

- `configmap.yaml`: Non-sensitive application configuration (environment variables, upload limits, service hostnames).
- `deployment.yaml`: FastAPI and Streamlit deployments featuring non-root security context, resource requests & limits, rolling update strategy (`maxSurge: 1`, `maxUnavailable: 0`), and startup, liveness, and readiness probes.
- `service.yaml`: Internal `ClusterIP` service for the FastAPI backend and external `NodePort` service for the Streamlit UI.
- `kustomization.yaml`: Kustomize orchestration file for unified cluster management.

---

## 🔒 Security & Reliability Specifications

1. **Non-Root Runtime User:** Container processes run under non-root UID `10001` with dropped Linux capabilities (`capabilities.drop: [ALL]`).
2. **Health Probes:**
   - **Startup Probe:** `/health/live` (handles TensorFlow model load latency on startup).
   - **Liveness Probe:** `/health/live` (monitors container process health every 10s).
   - **Readiness Probe:** `/health/ready` (verifies model weight readiness before receiving traffic).
3. **Resource Allocation:**
   - **FastAPI API:** Requests: 250m CPU / 512Mi RAM, Limits: 1000m CPU / 2Gi RAM.
   - **Streamlit UI:** Requests: 200m CPU / 256Mi RAM, Limits: 500m CPU / 1Gi RAM.

---

## 🚀 Deployment Instructions

### 1. Pre-requisites & Local Cluster Setup
Ensure `kubectl` and either `minikube` or `kind` are installed:
```bash
# Start local Kind cluster
kind create cluster --name lung-cancer-cluster

# Or start Minikube
minikube start
```

### 2. Build & Load Container Image
```bash
# Build local image
docker build -t explainable-lung-cancer-api:v1.0.0 .

# Load image into Kind cluster
kind load docker-image explainable-lung-cancer-api:v1.0.0 --name lung-cancer-cluster

# Or into Minikube
minikube image load explainable-lung-cancer-api:v1.0.0
```

### 3. Deploy via Kustomize
```bash
kubectl apply -k deploy/k8s/
```

### 4. Verify Rollout & Pod Status
```bash
# Check deployment status
kubectl rollout status deployment/fastapi-deployment
kubectl rollout status deployment/streamlit-deployment

# List running pods and services
kubectl get pods -l app=explainable-lung-cancer
kubectl get services -l app=explainable-lung-cancer
```

### 5. Access Application Services
- **Streamlit UI:** Access via `http://localhost:30501` (NodePort) or run `minikube service streamlit-service`.
- **FastAPI Endpoint:** Port-forward internal service to test endpoints:
```bash
kubectl port-forward svc/fastapi-service 8000:8000
curl http://localhost:8000/health/ready
```

---

## 📦 Model Artifact Storage Strategy

For production Kubernetes deployments:
1. **PersistentVolumeClaim (PVC):** Mount a shared NFS/EFS persistent volume containing `best_model.keras` to `/app/models`.
2. **Init Container / S3 Sync:** Use an Init Container to download model weights from Azure Blob Storage / AWS S3 into an `emptyDir` volume before container startup.
3. **Fallback Architecture:** When no weights file is mounted, the application dynamically builds the Xception transfer learning base architecture with pre-trained ImageNet weights.

---

## ⚠️ Testing Status Notice

> **Deployment Verification Disclaimer:**  
> The Kubernetes manifests in `deploy/k8s/` have been syntactically and structurally validated (`kubectl apply --dry-run=client`). Local container build and health probes have been verified via Docker.
