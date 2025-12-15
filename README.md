# Federated Learning MLOps Pipeline for E-commerce Customer Churn Prediction

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.14](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete end-to-end MLOps pipeline implementing **Federated Learning** for privacy-preserving customer churn prediction. This project demonstrates advanced ML engineering practices including automated CI/CD, model versioning, drift detection, and comprehensive monitoring.

---

## 🎯 Project Overview

This system predicts e-commerce customer churn using **Federated Learning** to ensure data privacy by training models locally on client data and aggregating updates to create a global model. The pipeline includes:

- ✅ **Federated Learning**: Privacy-preserving distributed training across simulated clients
- ✅ **MLflow Integration**: Automated model versioning, tracking, and intelligent promotion/rollback
- ✅ **Drift Detection**: Real-time data drift monitoring with automated retraining triggers
- ✅ **CI/CD Pipeline**: Jenkins automation with GitHub Actions support
- ✅ **Kubernetes Deployment**: Production-ready orchestration with Minikube
- ✅ **Comprehensive Monitoring**: Prometheus + Grafana + Alertmanager stack
- ✅ **Interactive Dashboards**: Streamlit UI for predictions and training metrics visualization

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Features](#-key-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Deployment Options](#-deployment-options)
- [Drift Detection & Retraining](#-drift-detection--automated-retraining)
- [Monitoring & Observability](#-monitoring--observability)
- [MLflow Model Management](#-mlflow-model-management)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CI/CD Pipeline                          │
│              (Jenkins / GitHub Actions)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼────────┐            ┌────────▼────────┐
│  Preprocessing  │            │  Fed. Training  │
│   (3 Clients)   │────────────│   (20 Rounds)   │
└────────┬────────┘            └────────┬────────┘
         │                               │
         │                      ┌────────▼────────┐
         │                      │  MLflow Server  │
         │                      │  Model Registry │
         │                      └────────┬────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
              ┌──────────▼──────────┐
              │   Model Serving     │
              │   (FastAPI + K8s)   │
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
┌────────▼────────┐ ┌───▼────┐ ┌───────▼────────┐
│   Prometheus    │ │Grafana │ │  Streamlit UI  │
│  (Metrics)      │ │(Dashbd)│ │  (Inference)   │
└────────┬────────┘ └────────┘ └────────────────┘
         │
┌────────▼────────┐
│  Alertmanager   │
│ (Drift Alerts)  │
└────────┬────────┘
         │
┌────────▼────────┐
│ Jenkins Trigger │
│  (Retraining)   │
└─────────────────┘
```

---

## ✨ Key Features

### 1. **Federated Learning**
- Simulates 3 independent clients with distributed data
- Privacy-preserving: raw data never leaves client nodes
- FedAvg algorithm for model aggregation
- Configurable number of rounds and clients

### 2. **MLflow Model Versioning**
- Automatic model registration with versioning
- **Intelligent Promotion Logic**:
  - Minimum accuracy threshold: **80%**
  - Automatic comparison with previous production model
  - Smart rollback if new model underperforms
  - Production/Staging/Archived lifecycle management
- Complete experiment tracking with metrics, params, and artifacts

### 3. **Data Drift Detection**
- Real-time Kolmogorov-Smirnov (KS) statistical tests
- Per-feature drift scoring with configurable thresholds
- Reference window: 100 samples
- Monitoring window: 50 samples
- Prometheus metrics for drift visualization
- Automated alerting via Alertmanager

### 4. **CI/CD Pipeline**
- **Jenkins Pipeline**:
  - Automated preprocessing, training, and deployment
  - Docker image building in Minikube
  - Health checks and test traffic generation
  - Parameterized builds (skip tests, force deploy)
- **GitHub Actions**:
  - Linting, testing, and validation
  - Artifact archiving

### 5. **Monitoring Stack**
- **Prometheus**: Metrics collection (5s scrape interval)
- **Grafana**: Pre-configured dashboards with 10+ panels
- **Alertmanager**: Webhook integration for drift alerts
- **Streamlit**: Interactive UI for predictions and training visualization

### 6. **Production-Ready Deployment**
- Kubernetes manifests for all services
- Persistent volume management for model data
- Rolling updates with zero downtime
- Resource limits and health checks
- Multi-replica API deployment (2 replicas)

---

## 📊 Dataset

This project uses the **E-Commerce Customer Churn Dataset** from Kaggle.

**Dataset Source**: [E-Commerce Data - Kaggle](https://www.kaggle.com/datasets/carrie1/ecommerce-data)

### Dataset Features (20 columns, 5630 rows):

**Target Variable:**
- `Churn`: Binary indicator (0 = No Churn, 1 = Churn)

**Numerical Features:**
- `Tenure`: Customer relationship duration (months)
- `WarehouseToHome`: Distance from warehouse to home (km)
- `HourSpendOnApp`: Daily app usage time
- `NumberOfDeviceRegistered`: Registered devices count
- `SatisfactionScore`: Customer satisfaction (1-5 scale)
- `NumberOfAddress`: Number of saved addresses
- `OrderAmountHikeFromlastYear`: YoY order amount increase (%)
- `CouponUsed`: Total coupons used
- `OrderCount`: Total orders placed
- `DaySinceLastOrder`: Days since last purchase
- `CashbackAmount`: Total cashback received ($)
- `Complain`: Binary complaint indicator

**Categorical Features:**
- `PreferredLoginDevice`: Mobile Phone / Computer
- `CityTier`: City classification (1, 2, 3)
- `PreferredPaymentMode`: Credit Card, Debit Card, UPI, etc.
- `Gender`: Male / Female
- `PreferedOrderCat`: Fashion, Electronics, Grocery, etc.
- `MaritalStatus`: Single, Married, Divorced

### Download Instructions:

1. Download from Kaggle: https://www.kaggle.com/datasets/carrie1/ecommerce-data
2. Place `E_Commerce_Dataset.xlsx` in the project root directory
3. Ensure the sheet name is `E Comm` (default)

---

## 🚀 Installation

### Prerequisites

- Python 3.10
- Docker & Docker Compose
- Minikube (for Kubernetes deployment)
- kubectl
- Git

### Setup Steps

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd mlops-federated-churn
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
- Get `E_Commerce_Dataset.xlsx` from Kaggle (link above)
- Place it in the project root

5. **Verify installation:**
```bash
python --version  # Should show Python 3.10.x
docker --version
kubectl version --client
minikube version
```

---

## ⚡ Quick Start

### Option 1: Local Development (Fastest)

```bash
# 1. Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 &

# 2. Preprocess data
python preprocess.py \
    --dataset E_Commerce_Dataset.xlsx \
    --output-folder preprocessed_data \
    --clients 3

# 3. Train federated model
python training_MLFlow.py \
    --num-rounds 20 \
    --batch-size 8

# 4. Start monitoring stack (Docker Compose)
docker-compose up -d

# 5. Launch Streamlit dashboard
streamlit run dashboard.py
```

**Access Services:**
- MLflow UI: http://localhost:5000
- API Docs: http://localhost:8000/docs
- Grafana: http://localhost:3000 (admin/root)
- Prometheus: http://localhost:9090
- Streamlit: http://localhost:8501

---

## 🐳 Deployment Options

### 1. Docker Compose (Development)

Perfect for local testing and development.

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f churn-api

# Stop all services
docker-compose down
```

**Services included:**
- `churn-api`: FastAPI serving (port 8000)
- `prometheus`: Metrics collection (port 9090)
- `grafana`: Dashboards (port 3000)
- `alertmanager`: Alert routing (port 9093)

---

### 2. Kubernetes with Minikube (Production-like)

Full production deployment with orchestration.

#### Initial Setup

```bash
# Start Minikube
minikube start --driver=docker --kubernetes-version=v1.28.0

# Verify cluster
kubectl cluster-info
kubectl get nodes
```

#### Deploy Services

```bash
# 1. Create namespace and resources
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml

# 2. Deploy monitoring stack
kubectl apply -f k8s/prometheus-rbac.yaml
kubectl apply -f k8s/prometheus-configmap.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/alertmanager-deployment.yaml

# 3. Deploy Grafana
kubectl apply -f k8s/grafana-datasource-configmap.yaml
kubectl apply -f k8s/grafana-dashboard-config.yaml
kubectl apply -f k8s/grafana-dashboard-json.yaml
kubectl apply -f k8s/grafana-deployment.yaml

# 4. Deploy MLflow (optional)
kubectl apply -f k8s/mlflow-deployment.yaml

# 5. Build Docker images in Minikube
eval $(minikube docker-env)
docker build -t churn-preprocess:latest -f Dockerfile.preprocess .
docker build -t churn-training:latest -f Dockerfile.training .
docker build -t churn-serving:latest -f Dockerfile.serving .

# 6. Run preprocessing and training
# (This is typically handled by Jenkins, but can be done manually)

# 7. Deploy API
kubectl apply -f k8s/api-deployment.yaml

# 8. Check deployment status
kubectl get pods -n churn-prediction
kubectl get services -n churn-prediction
```

#### Access Services

```bash
# Get service URLs
minikube service churn-api-service -n churn-prediction --url
minikube service grafana-service -n churn-prediction --url
minikube service prometheus-service -n churn-prediction --url

# Or use port-forwarding
kubectl port-forward -n churn-prediction svc/churn-api-service 8000:8000
kubectl port-forward -n churn-prediction svc/grafana-service 3000:3000
kubectl port-forward -n churn-prediction svc/prometheus-service 9090:9090
```

#### Copy Model Data to Kubernetes

```bash
# Create temporary pod for data transfer
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: data-copier
  namespace: churn-prediction
spec:
  containers:
  - name: copier
    image: busybox
    command: ['sh', '-c', 'sleep 3600']
    volumeMounts:
    - name: model-data
      mountPath: /data
  volumes:
  - name: model-data
    persistentVolumeClaim:
      claimName: model-data-pvc
  restartPolicy: Never
EOF

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/data-copier -n churn-prediction --timeout=60s

# Copy data
kubectl cp preprocessed_data churn-prediction/data-copier:/data/
kubectl cp federated_data churn-prediction/data-copier:/data/

# Verify
kubectl exec -n churn-prediction data-copier -- ls -la /data/preprocessed_data
kubectl exec -n churn-prediction data-copier -- ls -la /data/federated_data

# Clean up
kubectl delete pod data-copier -n churn-prediction
```

---

### 3. Jenkins CI/CD Pipeline (Automated)

Complete automation from code commit to production deployment.

#### Jenkins Setup

```bash
# 1. Start Jenkins (if not already running)
sudo systemctl start jenkins

# 2. Access Jenkins
# URL: http://localhost:8090
# Get initial admin password:
sudo cat /var/lib/jenkins/secrets/initialAdminPassword

# 3. Install required plugins:
# - Pipeline
# - Git
# - Docker Pipeline
# - Kubernetes
```

#### Create Pipeline Job

1. **New Item** → Enter name: `churn-prediction-pipeline`
2. Select **Pipeline** → Click OK
3. Configure:
   - **This project is parameterized**:
     - String: `DATASET_PATH` (default: `E_Commerce_Dataset.xlsx`)
     - Boolean: `SKIP_TESTS` (default: false)
     - Boolean: `FORCE_DEPLOY` (default: false)
   - **Pipeline**:
     - Definition: `Pipeline script from SCM`
     - SCM: Git
     - Repository URL: Your repo URL
     - Script Path: `Jenkinsfile`

#### Run Pipeline

```bash
# Trigger manually from Jenkins UI
# Or use Jenkins CLI:
java -jar jenkins-cli.jar -s http://localhost:8090/ build churn-prediction-pipeline

# Or trigger via webhook (from Alertmanager)
curl -X POST http://localhost:8090/job/churn-prediction-pipeline/build \
  --user $JENKINS_USER:$JENKINS_TOKEN
```

#### Pipeline Stages

The Jenkinsfile executes these stages automatically:

1. **Cleanup**: Clean workspace
2. **Checkout**: Pull latest code
3. **Setup**: Create directories, verify dataset
4. **Start MLflow Server**: Launch tracking server
5. **Build Docker Images**: Build all containers
6. **Data Preprocessing**: Run preprocessing in Docker
7. **Model Training**: Federated learning with MLflow
8. **Extract Metrics**: Validate model accuracy (threshold: 0.75)
9. **Verify Minikube**: Ensure Kubernetes cluster is ready
10. **Setup K8s Resources**: Create namespace, PVCs
11. **Copy Model Data**: Transfer to Kubernetes volumes
12. **Deploy to K8s**: Apply all manifests
13. **Health Check**: Verify API responsiveness
14. **Generate Test Traffic**: Create sample predictions

---

## 🔍 Drift Detection & Automated Retraining

### How Drift Detection Works

The system continuously monitors incoming predictions for data drift using the **Kolmogorov-Smirnov (KS) test**:

1. **Reference Window**: First 100 predictions establish baseline distribution
2. **Monitoring Window**: Next 50 predictions are compared to reference
3. **Statistical Test**: KS test performed on each numerical feature
4. **Threshold**: p-value < 0.05 indicates significant drift
5. **Alerting**: Prometheus metrics trigger Alertmanager alerts
6. **Retraining**: Alertmanager webhooks notify Jenkins to retrain

### Drift Detection Configuration

**In `app.py`:**
```python
DRIFT_THRESHOLD = 0.05      # p-value threshold for KS test
REFERENCE_WINDOW = 100      # Baseline sample size
MONITORING_WINDOW = 50      # Monitoring sample size
```

**In `k8s/prometheus-configmap.yaml`:**
```yaml
- alert: DataDriftDetected
  expr: increase(data_drift_detected_total[5m]) > 0
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "Data drift detected in feature {{ $labels.feature }}"
```

### Testing Drift Detection

#### Option 1: Using Drift Test Script

The `drift_test.sh` script generates synthetic drift automatically:

```bash
# 1. Get API URL (if using Minikube)
export API_URL=$(minikube service churn-api-service -n churn-prediction --url)

# 2. Make script executable
chmod +x drift_test.sh

# 3. Run drift test
./drift_test.sh

# Expected output:
# Phase 1: 100 normal predictions (reference data)
# Phase 2: 60 drifted predictions (monitoring data)
# 
# Features with expected drift:
# - Tenure (40-100 vs 1-60)
# - WarehouseToHome (40-80 vs 5-30)
# - HourSpendOnApp (6-12 vs 1-5)
# - SatisfactionScore (1-2 vs 1-5)
# - OrderCount (15-30 vs 1-10)
# - And more...
```

#### Option 2: Manual Testing

```bash
# 1. Generate 100 normal predictions (reference)
for i in {1..100}; do
  curl -X POST $API_URL/predict \
    -H "Content-Type: application/json" \
    -d '{
      "Tenure": 10.0,
      "PreferredLoginDevice": "Mobile Phone",
      "CityTier": 1,
      "WarehouseToHome": 15.0,
      "PreferredPaymentMode": "Credit Card",
      "Gender": "Male",
      "HourSpendOnApp": 3.0,
      "NumberOfDeviceRegistered": 3,
      "PreferedOrderCat": "Laptop & Accessory",
      "SatisfactionScore": 4,
      "MaritalStatus": "Single",
      "NumberOfAddress": 2,
      "Complain": 0,
      "OrderAmountHikeFromlastYear": 15.0,
      "CouponUsed": 1.0,
      "OrderCount": 5.0,
      "DaySinceLastOrder": 3.0,
      "CashbackAmount": 150.0
    }' -s > /dev/null
  sleep 0.1
done

# 2. Generate 60 drifted predictions (monitoring)
for i in {1..60}; do
  curl -X POST $API_URL/predict \
    -H "Content-Type: application/json" \
    -d '{
      "Tenure": 80.0,
      "PreferredLoginDevice": "Computer",
      "CityTier": 1,
      "WarehouseToHome": 60.0,
      "PreferredPaymentMode": "Credit Card",
      "Gender": "Male",
      "HourSpendOnApp": 10.0,
      "NumberOfDeviceRegistered": 8,
      "PreferedOrderCat": "Laptop & Accessory",
      "SatisfactionScore": 1,
      "MaritalStatus": "Single",
      "NumberOfAddress": 10,
      "Complain": 1,
      "OrderAmountHikeFromlastYear": 80.0,
      "CouponUsed": 15.0,
      "OrderCount": 25.0,
      "DaySinceLastOrder": 30.0,
      "CashbackAmount": 800.0
    }' -s > /dev/null
  sleep 0.1
done

# 3. Check drift status via API
curl $API_URL/drift-status

# 4. Manually trigger drift check
curl -X POST $API_URL/check-drift
```

### Monitoring Drift in Grafana

1. Open Grafana: http://localhost:3000
2. Navigate to **Churn Prediction API - Monitoring Dashboard**
3. Check panels:
   - **Data Drift Detection Status**: Table showing drift scores per feature
   - **Active Alerts**: Current firing alerts
   - **Drift Detection Events**: Timeline of drift occurrences

### Alertmanager Workflow

When drift is detected:

1. **Prometheus** evaluates alert rules every 30s
2. **Alertmanager** receives alerts and groups by feature
3. **Webhook** triggers Jenkins retraining job
4. **Jenkins** executes full pipeline:
   - Preprocesses data
   - Trains new model with MLflow
   - Validates model performance
   - Deploys to Kubernetes if improved

### Automated Retraining Configuration

**In `alertmanager.yml`:**
```yaml
receivers:
  - name: 'drift-webhook'
    webhook_configs:
      - url: 'http://churn-api:8000/webhook/drift-alert'
        send_resolved: true
```

**Webhook endpoint in `app.py`:**
```python
@app.post("/webhook/drift-alert")
async def receive_drift_alert(webhook: AlertWebhook):
    # Log alert details
    # Trigger Jenkins job (future implementation)
    # For now, just logs the alert
```

---

## 📊 Monitoring & Observability

### Prometheus Metrics

The FastAPI application exposes comprehensive metrics at `/metrics`:

**Prediction Metrics:**
- `predictions_total`: Total predictions made (Counter)
- `prediction_latency_seconds`: Prediction latency histogram
- `churn_predictions{label}`: Predictions by churn class (Counter)
- `prediction_errors_total`: Total prediction errors (Counter)

**Model Metrics:**
- `model_version`: Current model version in use (Gauge)

**Drift Metrics:**
- `data_drift_detected_total{feature}`: Drift detections per feature (Counter)
- `data_drift_score{feature}`: Current KS statistic per feature (Gauge)

**Example queries:**
```promql
# Predictions per second
rate(predictions_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))

# Drift score for Tenure feature
data_drift_score{feature="Tenure"}

# Alert if high error rate
rate(prediction_errors_total[5m]) > 0.05
```

### Grafana Dashboards

Pre-configured dashboard includes:

**Performance Panels:**
1. Predictions per Second (Stat)
2. Total Predictions (Stat)
3. Average Latency (Stat with thresholds)
4. Total Errors (Stat)
5. Prediction Rate Over Time (Time series)
6. Prediction Latency Percentiles (Time series: p50, p95, p99)

**ML Monitoring Panels:**
7. Data Drift Detection Status (Table)
8. Active Alerts (Table)
9. Drift Detection Events (Time series)
10. Churn Predictions Distribution (Pie chart)

**Dashboard Features:**
- 10s auto-refresh
- 1-hour default time range
- Drill-down capabilities
- Alert annotations

### Streamlit Dashboard

Interactive UI for users and ML engineers:

**Features:**
1. **Prediction Interface**:
   - Form-based input for all 18 features
   - Real-time churn probability calculation
   - Risk level indicators (Low/Medium/High)
   - Gauge chart visualization
   - Actionable recommendations

2. **Training Metrics Page** (`pages/metrics.py`):
   - Per-round evaluation metrics
   - Interactive Plotly charts
   - Smoothed curves with confidence intervals
   - Best performance summary
   - Metrics comparison table
   - CSV export functionality

3. **System Information**:
   - Model metadata display
   - Quick links to MLflow, Prometheus, Grafana
   - Service health status

**Launch Streamlit:**
```bash
streamlit run dashboard.py
```

Access at: http://localhost:8501

### MLflow Tracking UI

Complete experiment tracking and model registry:

**Features:**
- Experiment comparison
- Metric visualization (20 rounds of training)
- Parameter logging
- Artifact storage (models, plots, CSVs)
- Model versioning and stage transitions

**Access MLflow:**
```bash
# If not already running
mlflow server --host 0.0.0.0 --port 5000

# Open browser
http://localhost:5000
```

**Navigate to:**
- **Experiments** → `federated_churn_prediction`
- **Models** → `churn_prediction_model`

---

## 🎯 MLflow Model Management

### Model Versioning Workflow

Every training run creates a new model version with automatic lifecycle management:

#### 1. Model Registration

```python
# In training_MLFlow.py
model_info = mlflow.tensorflow.log_model(
    central_model, 
    artifact_path="model",
    registered_model_name="churn_prediction_model"
)
```

Each model version includes:
- Model weights and architecture
- Training metrics (accuracy, F1, precision, recall, ROC AUC)
- Hyperparameters (learning rates, batch size, num rounds)
- Artifacts (plots, CSVs, preprocessor)

#### 2. Intelligent Model Promotion

The system automatically decides whether to promote new models based on performance:

**Promotion Criteria:**

```python
MIN_ACCURACY_THRESHOLD = 0.80  # Minimum required accuracy
```

**Decision Logic:**

1. **Below Threshold (< 80% accuracy)**:
   - New version → **Archived**
   - Previous Production version → **Remains in Production**
   - Action: Keep using existing model

2. **Above Threshold & Better than Production**:
   - New version → **Promoted to Production**
   - Previous Production version → **Archived**
   - Action: Deploy new model

3. **Above Threshold but Worse than Production**:
   - New version → **Archived** (performance regression)
   - Previous Production version → **Remains in Production**
   - Action: **Automatic Rollback** - keep using existing model

4. **First Model**:
   - If accuracy ≥ 80% → **Promoted to Production**
   - If accuracy < 80% → **Archived**, manual intervention required

#### 3. Model Stages

- **Staging**: Models undergoing validation (manual)
- **Production**: Active model serving predictions
- **Archived**: Old or rejected models

### Using Models in Production

The FastAPI serving app automatically loads the Production model:

```python
# In app.py
def get_production_model():
    """Load the latest Production model from MLflow registry"""
    try:
        # Try Production stage first
        model_uri = f"models:/churn_prediction_model/Production"
        model = mlflow.tensorflow.load_model(model_uri)
        return model, version
    except:
        # Fallback to Staging
        model_uri = f"models:/churn_prediction_model/Staging"
        model = mlflow.tensorflow.load_model(model_uri)
        return model, version
```

### Reload Model After Retraining

```bash
# Trigger model reload via API endpoint
curl -X POST http://localhost:8000/reload-model

# Response includes new version and accuracy
{
  "status": "success",
  "new_version": "5",
  "accuracy": 0.9234,
  "message": "Model reloaded to version 5"
}
```

### Manual Model Management

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# List all versions
versions = client.search_model_versions("name='churn_prediction_model'")

# Promote specific version to Production
client.transition_model_version_stage(
    name="churn_prediction_model",
    version="3",
    stage="Production",
    archive_existing_versions=True
)

# Add tags
client.set_model_version_tag(
    name="churn_prediction_model",
    version="3",
    key="deployment_date",
    value="2024-12-15"
)

# Delete old versions
client.delete_model_version(
    name="churn_prediction_model",
    version="1"
)
```

### Model Metadata

Each model version stores:

```json
{
  "accuracy": 0.9234,
  "f1_score": 0.8945,
  "precision": 0.9012,
  "recall": 0.8878,
  "roc_auc": 0.9456,
  "num_clients": 3,
  "num_rounds": 20,
  "batch_size": 8,
  "client_lr": 0.01,
  "server_lr": 1.0
}
```

---

## 📡 API Documentation

### Base URL
- **Development**: `http://localhost:8000`
- **Kubernetes**: `http://<minikube-ip>:<node-port>`

### Endpoints

#### 1. Root Endpoint
```http
GET /
```

**Response:**
```json
{
  "message": "Federated Churn Prediction API",
  "version": "2.0.0",
  "model_version": "5",
  "mlflow_tracking_uri": "http://localhost:5000",
  "endpoints": {
    "predict": "/predict",
    "health": "/health",
    "metrics": "/metrics",
    "model-info": "/model-info",
    "reload-model": "/reload-model",
    "docs": "/docs"
  }
}
```

#### 2. Predict Churn
```http
POST /predict
```

**Request Body:**
```json
{
  "Tenure": 12.0,
  "PreferredLoginDevice": "Mobile Phone",
  "CityTier": 1,
  "WarehouseToHome": 15.0,
  "PreferredPaymentMode": "Credit Card",
  "Gender": "Male",
  "HourSpendOnApp": 3.5,
  "NumberOfDeviceRegistered": 3,
  "PreferedOrderCat": "Laptop & Accessory",
  "SatisfactionScore": 3,
  "MaritalStatus": "Single",
  "NumberOfAddress": 2,
  "Complain": 0,
  "OrderAmountHikeFromlastYear": 15.5,
  "CouponUsed": 2.0,
  "OrderCount": 5.0,
  "DaySinceLastOrder": 3.0,
  "CashbackAmount": 150.0
}
```

**Response:**
```json
{
  "churn_probability": 0.2345,
  "churn_prediction": 0,
  "risk_level": "Low",
  "latency_ms": 45.23,
  "model
