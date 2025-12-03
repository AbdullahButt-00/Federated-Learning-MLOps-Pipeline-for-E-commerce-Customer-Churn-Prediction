#!/usr/bin/env python
"""
Jenkins Trigger Service
Receives webhooks from Prometheus Alertmanager and triggers Jenkins pipeline
"""

from flask import Flask, request, jsonify
import requests
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
JENKINS_URL = os.getenv('JENKINS_URL', 'http://localhost:8090')
JENKINS_JOB = os.getenv('JENKINS_JOB', 'churn-prediction-pipeline')
JENKINS_USER = os.getenv('JENKINS_USER', 'admin')
JENKINS_TOKEN = os.getenv('JENKINS_TOKEN', '')  # Get from Jenkins user settings

# Drift dataset path
DRIFT_DATASET = os.getenv('DRIFT_DATASET', 'E_Commerce_Dataset_Drifted.xlsx')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/trigger-retraining', methods=['POST'])
def trigger_retraining():
    """
    Receive alert from Alertmanager and trigger Jenkins pipeline for retraining
    """
    try:
        # Parse alert payload
        data = request.get_json()
        logger.info(f"Received alert: {data}")
        
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        # Check if this is a drift alert
        alerts = data.get('alerts', [])
        drift_alerts = [
            alert for alert in alerts 
            if alert.get('labels', {}).get('alertname') == 'DataDriftDetected'
        ]
        
        if not drift_alerts:
            logger.info("No drift alerts found, ignoring")
            return jsonify({"status": "ignored", "reason": "no drift alerts"}), 200
        
        # Extract alert details
        alert = drift_alerts[0]
        feature = alert.get('labels', {}).get('feature', 'unknown')
        status = data.get('status', 'unknown')
        
        logger.warning(f"🚨 DRIFT ALERT: Feature={feature}, Status={status}")
        
        # Trigger Jenkins job for retraining
        if status == 'firing':
            logger.info(f"🔄 Triggering Jenkins retraining pipeline...")
            
            # Build Jenkins job URL
            jenkins_trigger_url = f"{JENKINS_URL}/job/{JENKINS_JOB}/buildWithParameters"
            
            # Parameters for Jenkins job
            params = {
                'DATASET_PATH': DRIFT_DATASET,  # Use drifted dataset
                'SKIP_TESTS': 'false',
                'FORCE_DEPLOY': 'false'
            }
            
            # Trigger Jenkins build
            if JENKINS_TOKEN:
                auth = (JENKINS_USER, JENKINS_TOKEN)
                response = requests.post(
                    jenkins_trigger_url,
                    params=params,
                    auth=auth,
                    timeout=10
                )
            else:
                logger.warning("No Jenkins token provided, attempting without auth")
                response = requests.post(
                    jenkins_trigger_url,
                    params=params,
                    timeout=10
                )
            
            if response.status_code in [200, 201]:
                logger.info(f"✓ Jenkins pipeline triggered successfully")
                return jsonify({
                    "status": "success",
                    "message": "Retraining pipeline triggered",
                    "jenkins_job": JENKINS_JOB,
                    "dataset": DRIFT_DATASET,
                    "feature": feature
                }), 200
            else:
                logger.error(f"❌ Failed to trigger Jenkins: {response.status_code} - {response.text}")
                return jsonify({
                    "status": "error",
                    "message": "Failed to trigger Jenkins",
                    "status_code": response.status_code
                }), 500
        
        else:
            logger.info(f"Alert status is '{status}', not triggering retraining")
            return jsonify({
                "status": "acknowledged",
                "message": f"Alert status: {status}"
            }), 200
    
    except Exception as e:
        logger.error(f"❌ Error processing alert: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/webhook/drift-alert', methods=['POST'])
def drift_alert_webhook():
    """
    Alternative endpoint compatible with existing alertmanager config
    """
    return trigger_retraining()

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Jenkins Trigger Service Starting")
    logger.info("="*60)
    logger.info(f"Jenkins URL: {JENKINS_URL}")
    logger.info(f"Jenkins Job: {JENKINS_JOB}")
    logger.info(f"Drift Dataset: {DRIFT_DATASET}")
    logger.info("="*60)
    
    app.run(host='0.0.0.0', port=5001, debug=False)