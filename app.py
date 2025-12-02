#!/usr/bin/env python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import time
import os
import logging
from scipy import stats
import json
from collections import deque
from threading import Lock
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== CONFIG =====================
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "preprocessed_data/preprocessor.pkl")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_REGISTRY_NAME = "churn_prediction_model"

# ===================== MLflow Setup =====================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# ===================== Model Loading =====================
def get_production_model():
    """
    Load the latest Production model from MLflow registry.
    Falls back to Staging if no Production model exists.
    """
    try:
        # Try to load from Production stage
        try:
            model_uri = f"models:/{MODEL_REGISTRY_NAME}/Production"
            logger.info(f"Loading Production model from: {model_uri}")
            model = mlflow.tensorflow.load_model(model_uri)
            
            # Get version info
            versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Production"])
            if versions:
                version = versions[0].version
                accuracy = versions[0].tags.get('accuracy', 'N/A')
                logger.info(f"✓ Loaded Production model version {version} (accuracy: {accuracy})")
                return model, version
            else:
                logger.info("✓ Loaded Production model")
                return model, "Production"
                
        except Exception as e:
            logger.warning(f"No Production model found, trying Staging: {e}")
            
            # Fallback to Staging
            try:
                model_uri = f"models:/{MODEL_REGISTRY_NAME}/Staging"
                logger.info(f"Loading Staging model from: {model_uri}")
                model = mlflow.tensorflow.load_model(model_uri)
                
                versions = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["Staging"])
                if versions:
                    version = versions[0].version
                    accuracy = versions[0].tags.get('accuracy', 'N/A')
                    logger.info(f"✓ Loaded Staging model version {version} (accuracy: {accuracy})")
                    return model, version
                else:
                    logger.info("✓ Loaded Staging model")
                    return model, "Staging"
                    
            except Exception as e2:
                logger.warning(f"No Staging model found, trying latest version: {e2}")
                
                # Final fallback: get latest version regardless of stage
                versions = client.search_model_versions(f"name='{MODEL_REGISTRY_NAME}'")
                if versions:
                    latest_version = max([int(v.version) for v in versions])
                    model_uri = f"models:/{MODEL_REGISTRY_NAME}/{latest_version}"
                    logger.info(f"Loading latest version {latest_version} from: {model_uri}")
                    model = mlflow.tensorflow.load_model(model_uri)
                    logger.info(f"✓ Loaded model version {latest_version}")
                    return model, str(latest_version)
                else:
                    raise RuntimeError("No model versions found in registry")
    
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        # Ultimate fallback to local file
        logger.warning("Falling back to local model file")
        model_path = os.getenv("MODEL_PATH", "federated_data/federated_churn_model.h5")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"✓ Loaded model from local file: {model_path}")
            return model, "local"
        else:
            raise RuntimeError(f"Could not load model from MLflow or local file: {e}")

# ===================== FastAPI App =====================
app = FastAPI(
    title="Federated Churn Prediction API",
    description="ML model serving with Prometheus metrics and MLflow integration",
    version="2.0.0"
)

# ===================== Prometheus Metrics =====================
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
CHURN_PREDICTIONS = Counter('churn_predictions', 'Churn predictions', ['label'])
ERROR_COUNTER = Counter('prediction_errors_total', 'Total prediction errors')
MODEL_VERSION_GAUGE = Gauge('model_version', 'Current model version in use')

# ===================== Drift Detection Setup =====================
DRIFT_THRESHOLD = 0.05  # p-value threshold for KS test
REFERENCE_WINDOW = 100  # Size of reference data window
MONITORING_WINDOW = 50  # Size of monitoring window for drift check

# Store recent predictions for drift detection
reference_data = deque(maxlen=REFERENCE_WINDOW)
monitoring_data = deque(maxlen=MONITORING_WINDOW)
data_lock = Lock()

# Prometheus metric for drift
DRIFT_DETECTED = Counter('data_drift_detected_total', 'Total drift detections', ['feature'])
DRIFT_SCORE = Gauge('data_drift_score', 'Current drift score (KS statistic)', ['feature'])

# ===================== Load Model & Preprocessor =====================
try:
    logger.info("=" * 60)
    logger.info("Initializing Churn Prediction API")
    logger.info("=" * 60)
    
    # Load model from MLflow
    model, current_model_version = get_production_model()
    
    # Update Prometheus gauge
    try:
        MODEL_VERSION_GAUGE.set(int(current_model_version))
    except ValueError:
        MODEL_VERSION_GAUGE.set(0)  # If version is not a number
    
    logger.info(f"Loading preprocessor from: {PREPROCESSOR_PATH}")
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    logger.info("✓ Preprocessor loaded successfully")
    
    logger.info("=" * 60)
    logger.info(f"✓ API Ready - Using Model Version: {current_model_version}")
    logger.info("=" * 60)
    
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise RuntimeError(f"Required file not found: {e}")
except Exception as e:
    logger.error(f"Error loading model/preprocessor: {e}")
    raise RuntimeError(f"Failed to load model/preprocessor: {e}")

# ===================== Pydantic Models =====================
class CustomerData(BaseModel):
    Tenure: float
    PreferredLoginDevice: str
    CityTier: int
    WarehouseToHome: float
    PreferredPaymentMode: str
    Gender: str
    HourSpendOnApp: float
    NumberOfDeviceRegistered: int
    PreferedOrderCat: str
    SatisfactionScore: int
    MaritalStatus: str
    NumberOfAddress: int
    Complain: int
    OrderAmountHikeFromlastYear: float
    CouponUsed: float
    OrderCount: float
    DaySinceLastOrder: float
    CashbackAmount: float
    
    class Config:
        schema_extra = {
            "example": {
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
        }

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    latency_ms: float
    model_version: str
    
class AlertWebhook(BaseModel):
    """Webhook payload from Alertmanager"""
    version: str
    groupKey: str
    status: str
    alerts: list
    
def detect_drift(feature_name, reference, monitoring):
    """
    Perform Kolmogorov-Smirnov test to detect drift
    Returns: (is_drift, ks_statistic, p_value)
    """
    if len(reference) < 30 or len(monitoring) < 30:
        return False, 0.0, 1.0
    
    ks_statistic, p_value = stats.ks_2samp(reference, monitoring)
    is_drift = p_value < DRIFT_THRESHOLD
    
    return is_drift, ks_statistic, p_value

def check_all_features_drift(reference_df, monitoring_df):
    """
    Check drift across all numeric features
    Returns dict of features with drift detected
    """
    drift_detected = {}
    
    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in monitoring_df.columns:
            is_drift, ks_stat, p_val = detect_drift(
                col,
                reference_df[col].values,
                monitoring_df[col].values
            )
            
            # Update Prometheus metrics
            DRIFT_SCORE.labels(feature=col).set(ks_stat)
            
            if is_drift:
                DRIFT_DETECTED.labels(feature=col).inc()
                drift_detected[col] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_val)
                }
                logger.warning(f"⚠️ DRIFT DETECTED in {col}: KS={ks_stat:.4f}, p={p_val:.4f}")
    
    return drift_detected

# ===================== API Endpoints =====================
@app.get("/")
async def root():
    return {
        "message": "Federated Churn Prediction API",
        "version": "2.0.0",
        "model_version": str(current_model_version),
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "model-info": "/model-info",
            "reload-model": "/reload-model",
            "docs": "/docs"
        }
    }

@app.post("/webhook/drift-alert")
async def receive_drift_alert(webhook: AlertWebhook):
    """
    Receive drift alerts from Alertmanager
    This will later trigger retraining
    """
    logger.warning(f"🚨 ALERT RECEIVED: {webhook.status}")
    
    for alert in webhook.alerts:
        alert_name = alert.get('labels', {}).get('alertname')
        feature = alert.get('labels', {}).get('feature', 'unknown')
        
        logger.warning(f"Alert: {alert_name}, Feature: {feature}")
        logger.warning(f"Description: {alert.get('annotations', {}).get('description')}")
        
        # TODO: Later, trigger retraining here
        # For now, just log
        if alert_name == 'DataDriftDetected':
            logger.critical(f"⚠️ RETRAINING NEEDED for feature: {feature}")
    
    return {"status": "received"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(data: CustomerData):
    """
    Predict customer churn probability
    """
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Store raw input for drift detection
        with data_lock:
            if len(reference_data) < REFERENCE_WINDOW:
                reference_data.append(df.copy())
            else:
                monitoring_data.append(df.copy())
                
                # Check for drift when monitoring window is full
                if len(monitoring_data) == MONITORING_WINDOW:
                    ref_df = pd.concat(list(reference_data), ignore_index=True)
                    mon_df = pd.concat(list(monitoring_data), ignore_index=True)
                    
                    drift_results = check_all_features_drift(ref_df, mon_df)
                    
                    if drift_results:
                        logger.warning(f"🚨 Drift detected in {len(drift_results)} features")
        
        # Preprocess
        X_transformed = preprocessor.transform(df)
        
        # Predict
        prediction_prob = float(model.predict(X_transformed, verbose=0)[0][0])
        prediction_label = int(prediction_prob >= 0.5)
        
        # Determine risk level
        if prediction_prob > 0.7:
            risk_level = "High"
        elif prediction_prob > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Update Prometheus metrics
        PREDICTION_COUNTER.inc()
        CHURN_PREDICTIONS.labels(label=str(prediction_label)).inc()
        
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        
        logger.info(f"Prediction: {prediction_label}, Probability: {prediction_prob:.3f}, Latency: {latency*1000:.2f}ms")
        
        return {
            "churn_probability": round(prediction_prob, 4),
            "churn_prediction": prediction_label,
            "risk_level": risk_level,
            "latency_ms": round(latency * 1000, 2),
            "model_version": str(current_model_version)
        }
    
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    try:
        # Quick model test
        test_input = np.zeros((1, model.input_shape[1]))
        _ = model.predict(test_input, verbose=0)
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "preprocessor_loaded": True,
            "model_version": str(current_model_version)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/model-info")
async def model_info():
    """
    Get model metadata
    """
    # Try to get accuracy from MLflow
    accuracy = None
    try:
        if current_model_version != "local":
            versions = client.search_model_versions(
                f"name='{MODEL_REGISTRY_NAME}' and version='{current_model_version}'"
            )
            if versions and 'accuracy' in versions[0].tags:
                accuracy = float(versions[0].tags['accuracy'])
    except:
        pass
    
    return {
        "model_registry_name": MODEL_REGISTRY_NAME,
        "current_version": str(current_model_version),
        "accuracy": accuracy,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "model_type": "TensorFlow/Keras",
        "preprocessor_path": PREPROCESSOR_PATH,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI
    }

@app.post("/reload-model")
async def reload_model():
    """
    Reload the model from MLflow registry (useful after retraining)
    """
    global model, current_model_version
    
    try:
        logger.info("Reloading model from MLflow registry...")
        new_model, new_version = get_production_model()
        
        # Update global variables
        model = new_model
        current_model_version = new_version
        
        # Update Prometheus gauge
        try:
            MODEL_VERSION_GAUGE.set(int(current_model_version))
        except ValueError:
            MODEL_VERSION_GAUGE.set(0)
        
        # Get accuracy if available
        accuracy = None
        try:
            if current_model_version != "local":
                versions = client.search_model_versions(
                    f"name='{MODEL_REGISTRY_NAME}' and version='{current_model_version}'"
                )
                if versions and 'accuracy' in versions[0].tags:
                    accuracy = float(versions[0].tags['accuracy'])
        except:
            pass
        
        logger.info(f"✓ Model reloaded: version {current_model_version}")
        
        return {
            "status": "success",
            "new_version": str(current_model_version),
            "accuracy": accuracy,
            "message": f"Model reloaded to version {current_model_version}"
        }
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")
    
@app.get("/drift-status")
async def drift_status():
    """
    Get current drift detection status
    """
    with data_lock:
        return {
            "reference_samples": len(reference_data),
            "monitoring_samples": len(monitoring_data),
            "drift_threshold": DRIFT_THRESHOLD,
            "monitoring_active": len(monitoring_data) > 0
        }
        
@app.post("/check-drift")
async def manual_drift_check():
    """
    Manually trigger drift detection
    """
    with data_lock:
        if len(reference_data) < 30 or len(monitoring_data) < 30:
            raise HTTPException(
                status_code=400,
                detail="Not enough data for drift detection"
            )
        
        ref_df = pd.concat(list(reference_data), ignore_index=True)
        mon_df = pd.concat(list(monitoring_data), ignore_index=True)
        
        drift_results = check_all_features_drift(ref_df, mon_df)
        
        return {
            "drift_detected": len(drift_results) > 0,
            "features_with_drift": drift_results,
            "total_features_checked": len(ref_df.select_dtypes(include=[np.number]).columns)
        }

# ===================== Prometheus Metrics Endpoint =====================
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ===================== Main =====================
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Federated Churn Prediction API...")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )