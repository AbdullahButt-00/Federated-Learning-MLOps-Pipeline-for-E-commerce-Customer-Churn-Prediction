#!/bin/bash

echo "🚀 Deploying Churn Prediction System to Kubernetes..."

# Build Docker images (update with your registry)
REGISTRY="your-registry"  # e.g., docker.io/username or gcr.io/project-id

echo "📦 Building Docker images..."
docker build -f Dockerfile.preprocess -t $REGISTRY/churn-preprocess:latest .
docker build -f Dockerfile.training -t $REGISTRY/churn-training:latest .
docker build -f Dockerfile.serving -t $REGISTRY/churn-serving:latest .

echo "📤 Pushing images to registry..."
docker push $REGISTRY/churn-preprocess:latest
docker push $REGISTRY/churn-training:latest
docker push $REGISTRY/churn-serving:latest

echo "🔧 Creating namespace..."
kubectl apply -f k8s/namespace.yaml

echo "📝 Creating ConfigMaps..."
kubectl apply -f k8s/prometheus-configmap.yaml
kubectl apply -f k8s/grafana-datasource-configmap.yaml
kubectl apply -f k8s/grafana-dashboard-config.yaml
kubectl apply -f k8s/grafana-dashboard-json.yaml

echo "💾 Creating Persistent Volume Claims..."
kubectl apply -f k8s/pvc.yaml

echo "📊 Deploying MLflow..."
kubectl apply -f k8s/mlflow-deployment.yaml

echo "⏳ Waiting for MLflow to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/mlflow -n churn-prediction

echo "🔄 Running preprocessing job..."
kubectl apply -f k8s/preprocess-job.yaml
kubectl wait --for=condition=complete --timeout=600s job/preprocess-job -n churn-prediction

echo "🎯 Running training job..."
kubectl apply -f k8s/training-job.yaml
kubectl wait --for=condition=complete --timeout=1800s job/training-job -n churn-prediction

echo "🚀 Deploying API service..."
kubectl apply -f k8s/api-deployment.yaml

echo "📈 Deploying Prometheus..."
kubectl apply -f k8s/prometheus-rbac.yaml
kubectl apply -f k8s/prometheus-deployment.yaml

echo "📊 Deploying Grafana..."
kubectl apply -f k8s/grafana-deployment.yaml

echo "✅ Deployment complete!"
