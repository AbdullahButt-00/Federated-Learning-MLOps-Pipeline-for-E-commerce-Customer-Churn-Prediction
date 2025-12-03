#!/usr/bin/env python
import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import subprocess

def get_minikube_service_url(service_name, namespace="churn-prediction"):
    """Get minikube service URL"""
    try:
        result = subprocess.run(
            ["minikube", "service", service_name, "-n", namespace, "--url"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except:
        return None

API_URL = get_minikube_service_url("churn-api-service") or "http://localhost:8000"
PROMETHEUS_URL = get_minikube_service_url("prometheus-service") or "http://localhost:9090"
GRAFANA_URL = get_minikube_service_url("grafana-service") or "http://localhost:3000"
MLFLOW_URL = "http://localhost:5000"

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3em;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# ===================== TITLE =====================
st.title("🔮 Customer Churn Prediction Dashboard")
st.markdown("---")

# ===================== CHECK API HEALTH =====================
try:
    health_response = requests.get(f"{API_URL}/health", timeout=2)
    if health_response.status_code == 200:
        st.success("✅ API is healthy and ready")
    else:
        st.error("❌ API is not responding correctly")
except:
    st.error(f"❌ Cannot connect to API. Make sure it's running on {API_URL}")
    st.info("Check your Kubernetes service status")
    st.stop()

# ===================== SIDEBAR =====================
st.sidebar.title("🎛️ Control Panel")

# Add navigation hint
st.sidebar.info("💡 **Tip:** Check out the **Training Metrics** page for detailed model performance!")

# ===================== PREDICTION INPUT =====================
with st.sidebar.expander("🔮 **Make Prediction**", expanded=True):
    st.markdown("### Customer Information")
    
    with st.form("prediction_form"):
        st.subheader("Basic Info")
        
        tenure = st.number_input(
            "Tenure (months)", 
            min_value=0, 
            max_value=100, 
            value=10,
            help="How long the customer has been with the company"
        )
        
        gender = st.selectbox(
            "Gender", 
            ["Male", "Female"]
        )
        
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"]
        )
        
        city_tier = st.selectbox(
            "City Tier", 
            [1, 2, 3],
            help="1 = Metro, 2 = Tier-1, 3 = Tier-2"
        )
        
        st.markdown("---")
        st.subheader("Device & App Usage")
        
        preferred_login_device = st.selectbox(
            "Preferred Login Device", 
            ["Mobile Phone", "Computer"]
        )
        
        number_of_device_registered = st.number_input(
            "Number of Devices Registered", 
            min_value=1, 
            max_value=10, 
            value=3
        )
        
        hour_spend_on_app = st.number_input(
            "Hours Spent on App (daily)", 
            min_value=0.0, 
            max_value=24.0, 
            value=3.0,
            step=0.5
        )
        
        st.markdown("---")
        st.subheader("Shopping Behavior")
        
        prefered_order_cat = st.selectbox(
            "Preferred Order Category",
            ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"]
        )
        
        preferred_payment_mode = st.selectbox(
            "Preferred Payment Mode",
            ["Credit Card", "Debit Card", "Cash on Delivery", "UPI", "E wallet"]
        )
        
        order_count = st.number_input(
            "Total Orders", 
            min_value=0.0, 
            max_value=100.0, 
            value=5.0,
            step=1.0
        )
        
        day_since_last_order = st.number_input(
            "Days Since Last Order", 
            min_value=0.0, 
            max_value=100.0, 
            value=3.0,
            step=1.0
        )
        
        order_amount_hike_from_last_year = st.number_input(
            "Order Amount Hike (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=15.0,
            step=1.0
        )
        
        st.markdown("---")
        st.subheader("Satisfaction & Support")
        
        satisfaction_score = st.slider(
            "Satisfaction Score", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="1 = Very Satisfied, 5 = Very Unsatisfied"
        )
        
        complain = st.selectbox(
            "Has Complained?", 
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        
        st.markdown("---")
        st.subheader("Offers & Rewards")
        
        coupon_used = st.number_input(
            "Coupons Used", 
            min_value=0.0, 
            max_value=50.0, 
            value=1.0,
            step=1.0
        )
        
        cashback_amount = st.number_input(
            "Total Cashback Received", 
            min_value=0.0, 
            max_value=1000.0, 
            value=100.0,
            step=10.0
        )
        
        st.markdown("---")
        st.subheader("Location")
        
        warehouse_to_home = st.number_input(
            "Distance: Warehouse to Home (km)", 
            min_value=0.0, 
            max_value=200.0, 
            value=15.0,
            step=1.0
        )
        
        number_of_address = st.number_input(
            "Number of Addresses", 
            min_value=1, 
            max_value=20, 
            value=2
        )
        
        st.markdown("---")
        
        # Submit button
        submit_button = st.form_submit_button("🔮 Predict Churn Risk")

# ===================== SYSTEM INFO =====================
with st.sidebar.expander("ℹ️ **System Info**", expanded=False):
    st.markdown("### API Status")
    
    try:
        model_info = requests.get(f"{API_URL}/model-info", timeout=2).json()
        st.success("✅ Model Loaded")
        
        st.markdown("**Model Path:**")
        st.code(model_info.get('model_path', 'N/A'), language="text")
        
        st.markdown("**Input Shape:**")
        st.text(model_info.get('input_shape', 'N/A'))
        
        st.markdown("**Model Type:**")
        st.text(model_info.get('model_type', 'N/A'))
        
    except:
        st.warning("⚠️ Could not fetch model info")
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown(f"- [API Docs]({API_URL}/docs)")
    st.markdown(f"- [Prometheus Metrics]({API_URL}/metrics)")
    st.markdown(f"- [MLflow UI]({MLFLOW_URL})")
    st.markdown(f"- [Prometheus]({PROMETHEUS_URL})")
    st.markdown(f"- [Grafana]({GRAFANA_URL})")

# ===================== MAIN AREA =====================
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Prediction Results")
    
    if submit_button:
        # Prepare data for API
        customer_data = {
            "Tenure": tenure,
            "PreferredLoginDevice": preferred_login_device,
            "CityTier": city_tier,
            "WarehouseToHome": warehouse_to_home,
            "PreferredPaymentMode": preferred_payment_mode,
            "Gender": gender,
            "HourSpendOnApp": hour_spend_on_app,
            "NumberOfDeviceRegistered": number_of_device_registered,
            "PreferedOrderCat": prefered_order_cat,
            "SatisfactionScore": satisfaction_score,
            "MaritalStatus": marital_status,
            "NumberOfAddress": number_of_address,
            "Complain": complain,
            "OrderAmountHikeFromlastYear": order_amount_hike_from_last_year,
            "CouponUsed": coupon_used,
            "OrderCount": order_count,
            "DaySinceLastOrder": day_since_last_order,
            "CashbackAmount": cashback_amount
        }
        
        # Show loading spinner
        with st.spinner("🔄 Making prediction..."):
            try:
                # Call API
                response = requests.post(
                    f"{API_URL}/predict",
                    json=customer_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results in metrics
                    st.success("✅ Prediction completed successfully!")
                    
                    # Create 3 columns for metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        churn_prob = result['churn_probability']
                        st.metric(
                            "Churn Probability",
                            f"{churn_prob:.1%}",
                            delta=f"{churn_prob - 0.5:.1%}" if churn_prob > 0.5 else f"{0.5 - churn_prob:.1%}",
                            delta_color="inverse"
                        )
                    
                    with metric_col2:
                        prediction = "⚠️ WILL CHURN" if result['churn_prediction'] == 1 else "✅ WILL STAY"
                        st.metric(
                            "Prediction",
                            prediction,
                            delta=None
                        )
                    
                    with metric_col3:
                        risk_level = result['risk_level']
                        risk_color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
                        st.metric(
                            "Risk Level",
                            f"{risk_color} {risk_level}",
                            delta=None
                        )
                    
                    st.markdown("---")
                    
                    # Probability gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=churn_prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Churn Risk", 'font': {'size': 24}},
                        delta={'reference': 50, 'increasing': {'color': "red"}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 40], 'color': '#90EE90'},
                                {'range': [40, 70], 'color': '#FFD700'},
                                {'range': [70, 100], 'color': '#FF6B6B'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance metrics
                    st.info(f"⚡ Prediction completed in {result['latency_ms']:.2f}ms")
                    
                    # Recommendations
                    st.markdown("---")
                    st.subheader("💡 Recommendations")
                    
                    if result['churn_prediction'] == 1:
                        st.error("⚠️ **High Churn Risk Detected!**")
                        st.markdown("""
                        **Recommended Actions:**
                        - 🎁 Offer personalized discount or loyalty rewards
                        - 📞 Proactive customer service outreach
                        - 💳 Provide exclusive offers based on preferences
                        - 📧 Send re-engagement email campaign
                        - 🎯 Target with retention-focused ads
                        """)
                    else:
                        st.success("✅ **Low Churn Risk - Customer Likely to Stay**")
                        st.markdown("""
                        **Recommended Actions:**
                        - 🌟 Continue providing excellent service
                        - 📈 Upsell/cross-sell opportunities
                        - 💬 Request feedback and reviews
                        - 🎊 Include in referral programs
                        """)
                    
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.json(response.json())
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timeout. API took too long to respond.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Connection error. Make sure the API is running.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    else:
        # Show placeholder when no prediction yet
        st.info("👈 Fill in the customer details on the left and click 'Predict Churn Risk'")
        
        # Show example visualization
        st.subheader("📈 Sample Churn Risk Distribution")
        sample_data = pd.DataFrame({
            'Risk Level': ['Low', 'Medium', 'High'],
            'Count': [450, 300, 150]
        })
        fig = px.pie(
            sample_data, 
            values='Count', 
            names='Risk Level',
            color='Risk Level',
            color_discrete_map={'Low': '#90EE90', 'Medium': '#FFD700', 'High': '#FF6B6B'}
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("ℹ️ About")
    
    st.markdown("""
    ### Churn Prediction Model
    
    This dashboard uses a **Federated Learning** model trained on customer behavior data to predict churn risk.
    
    **Features Used:**
    - Customer demographics
    - App usage patterns
    - Shopping behavior
    - Satisfaction metrics
    - Payment preferences
    
    **Model Performance:**
    - Check **Training Metrics** page (in sidebar)
    - Training: Federated Learning
    - Framework: TensorFlow
    
    ---
    
    ### Risk Levels
    
    🟢 **Low (0-40%):** Customer is likely to stay
    
    🟡 **Medium (40-70%):** Monitor customer engagement
    
    🔴 **High (70-100%):** Immediate retention action needed
    
    ---
    
    ### Quick Actions
    """)
    
    if st.button("📊 Open MLflow UI"):
        st.markdown(f"Open MLflow at: {MLFLOW_URL}")
        st.code(MLFLOW_URL, language="text")
    
    if st.button("📈 Open Prometheus"):
        st.markdown(f"Open Prometheus at: {PROMETHEUS_URL}")
        st.code(PROMETHEUS_URL, language="text")
    
    if st.button("📉 Open Grafana"):
        st.markdown(f"Open Grafana at: {GRAFANA_URL}")
        st.code(GRAFANA_URL, language="text")
        st.info("Default credentials: admin/admin")

# ===================== FOOTER =====================
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center'>
        <p>🔮 Federated Churn Prediction Dashboard | Built with Streamlit & FastAPI</p>
        <p style='font-size: 12px; color: gray;'>
            Monitoring: <a href="{PROMETHEUS_URL}" target="_blank">Prometheus</a> | 
            <a href="{GRAFANA_URL}" target="_blank">Grafana</a> | 
            <a href="{MLFLOW_URL}" target="_blank">MLflow</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)