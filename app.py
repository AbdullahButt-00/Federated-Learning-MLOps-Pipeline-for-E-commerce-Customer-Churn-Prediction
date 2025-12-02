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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== CONFIG =====================
MODEL_PATH = os.getenv("MODEL_PATH", "federated_data/federated_churn_model.h5")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "preprocessed_data/preprocessor.pkl")

# ===================== FastAPI App =====================
app = FastAPI(
    title="Federated Churn Prediction API",
    description="ML model serving with Prometheus metrics",
    version="1.0.0"
)

# ===================== Prometheus Metrics =====================
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
CHURN_PREDICTIONS = Counter('churn_predictions', 'Churn predictions', ['label'])
ERROR_COUNTER = Counter('prediction_errors_total', 'Total prediction errors')

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
    logger.info(f"Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("✓ Model loaded successfully")
    
    logger.info(f"Loading preprocessor from: {PREPROCESSOR_PATH}")
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    logger.info("✓ Preprocessor loaded successfully")
    
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
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
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
            "latency_ms": round(latency * 1000, 2)
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
            "preprocessor_loaded": True
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/model-info")
async def model_info():
    """
    Get model metadata
    """
    return {
        "model_path": MODEL_PATH,
        "input_shape": model.input_shape,
        "output_shape": model.output_shape,
        "model_type": "TensorFlow/Keras",
        "preprocessor_path": PREPROCESSOR_PATH
    }
    
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
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Preprocessor path: {PREPROCESSOR_PATH}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )