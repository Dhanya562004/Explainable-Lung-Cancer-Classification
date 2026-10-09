# ☁️ Azure Container Apps Deployment Guide

This guide provides instructions for deploying the **Explainable Lung Cancer Classification API and UI** to **Azure Container Apps (ACA)** with **Azure Container Registry (ACR)**.

---

## 🔒 Security & OIDC Workload Identity

This deployment pipeline uses **GitHub OIDC Federated Identity** to authenticate securely with Azure without committing permanent service principal secret keys.

### 🔑 Required GitHub Secrets
Configure the following secrets in your GitHub Repository under `Settings -> Secrets and variables -> Actions`:

| Secret Name | Description |
| :--- | :--- |
| `AZURE_CLIENT_ID` | Application (Client) ID of the Azure AD App Registration |
| `AZURE_TENANT_ID` | Azure Active Directory Tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Target Azure Subscription ID |

---

## 💰 Resource Requirements & Estimated Azure Costs

| Resource Type | SKU / Spec | Estimated Cost (USD) |
| :--- | :--- | :--- |
| **Azure Container Registry (ACR)** | Basic SKU | ~$5.00 / month |
| **Azure Container Apps (FastAPI)** | 0.5 vCPU, 1.0 GiB RAM (min 1 replica) | ~$12.00 / month |
| **Azure Container Apps (Streamlit)** | 0.25 vCPU, 0.5 GiB RAM (min 1 replica) | ~$6.00 / month |
| **Log Analytics Workspace** | Pay-as-you-go (First 5GB free) | ~$0.00 - $3.00 / month |
| **Total Estimated Cost** | — | **~$23.00 - $26.00 / month** |

---

## 🚀 Manual Command-Line Deployment via Azure CLI

```bash
# 1. Log in to Azure
az login

# 2. Set Active Subscription
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# 3. Create Resource Group
az group create --name rg-lungcancer-prod --location eastus

# 4. Deploy Infrastructure via Bicep
az deployment group create \
  --resource-group rg-lungcancer-prod \
  --template-file deploy/azure/azure-container-apps.bicep \
  --parameters namePrefix=lungcancer imageTag=v1.0.0
```

---

## ⚠️ Cloud Deployment Disclaimer

> **Verification Notice:**  
> The Bicep infrastructure templates (`azure-container-apps.bicep`) and GitHub Action workflow (`deploy-azure.yml`) are fully authored and ready for cloud deployment. Active Azure cloud provisioning requires user Azure credentials and subscription activation.
