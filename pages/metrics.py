import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Training Metrics",
    page_icon="📊",
    layout="wide"
)

# ===================== CONFIG =====================
MLFLOW_URL = "http://localhost:5000"
METRICS_CSV = "federated_data/round_evaluation/per_round_metrics.csv"

# ===================== CUSTOM CSS =====================
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0px;
    }
    .big-metric {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# ===================== TITLE =====================
st.title("📊 Federated Learning Training Metrics")
st.markdown("### Comprehensive Model Performance Analysis")
st.markdown("---")

# ===================== LOAD METRICS =====================
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_metrics():
    """Load metrics from CSV file"""
    if os.path.exists(METRICS_CSV):
        return pd.read_csv(METRICS_CSV)
    return None

metrics_df = load_metrics()

# ===================== CHECK IF DATA EXISTS =====================
if metrics_df is None or len(metrics_df) == 0:
    st.error("❌ No training metrics found!")
    st.info("""
    **How to generate metrics:**
    1. Run the federated training script:
    ```bash
    python federated_training_with_mlflow.py
    ```
    2. Wait for training to complete
    3. Refresh this page
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh Page"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        if st.button("📂 Check File Location"):
            st.code(f"Expected location: {os.path.abspath(METRICS_CSV)}")
    
    st.stop()

# ===================== METRICS OVERVIEW =====================
st.success(f"✅ Successfully loaded {len(metrics_df)} training rounds")

# Get latest metrics
latest = metrics_df.iloc[-1]
total_rounds = len(metrics_df)

# ===================== TOP LEVEL METRICS =====================
st.markdown("## 🎯 Final Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        "Accuracy",
        f"{latest['eval_accuracy']:.4f}",
        delta=f"+{latest['eval_accuracy'] - metrics_df.iloc[0]['eval_accuracy']:.4f}" if len(metrics_df) > 1 else None
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        "Precision",
        f"{latest['eval_precision']:.4f}",
        delta=f"+{latest['eval_precision'] - metrics_df.iloc[0]['eval_precision']:.4f}" if len(metrics_df) > 1 else None
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        "Recall",
        f"{latest['eval_recall']:.4f}",
        delta=f"+{latest['eval_recall'] - metrics_df.iloc[0]['eval_recall']:.4f}" if len(metrics_df) > 1 else None
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        "F1 Score",
        f"{latest['eval_f1']:.4f}",
        delta=f"+{latest['eval_f1'] - metrics_df.iloc[0]['eval_f1']:.4f}" if len(metrics_df) > 1 else None
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        "ROC AUC",
        f"{latest['eval_roc_auc']:.4f}",
        delta=f"+{latest['eval_roc_auc'] - metrics_df.iloc[0]['eval_roc_auc']:.4f}" if len(metrics_df) > 1 else None
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ===================== BEST PERFORMANCE SUMMARY =====================
st.markdown("## 🏆 Best Performance Achieved")

col1, col2, col3 = st.columns(3)

with col1:
    best_acc_round = metrics_df['eval_accuracy'].idxmax() + 1
    best_acc_value = metrics_df['eval_accuracy'].max()
    st.info(f"""
    **Best Accuracy**  
    Round {best_acc_round}: **{best_acc_value:.4f}**
    """)

with col2:
    best_f1_round = metrics_df['eval_f1'].idxmax() + 1
    best_f1_value = metrics_df['eval_f1'].max()
    st.info(f"""
    **Best F1 Score**  
    Round {best_f1_round}: **{best_f1_value:.4f}**
    """)

with col3:
    best_auc_round = metrics_df['eval_roc_auc'].idxmax() + 1
    best_auc_value = metrics_df['eval_roc_auc'].max()
    st.info(f"""
    **Best ROC AUC**  
    Round {best_auc_round}: **{best_auc_value:.4f}**
    """)

st.markdown("---")

# ===================== INDIVIDUAL METRIC PLOTS =====================
st.markdown("## 📈 Training Progress - Individual Metrics")

# Create tabs for each metric
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Accuracy", "🎯 Precision", "🔍 Recall", "⚖️ F1 Score", "📉 ROC AUC"])

with tab1:
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=metrics_df['round'],
        y=metrics_df['eval_accuracy'],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig_acc.update_layout(
        title="Accuracy Progress Across Rounds",
        xaxis_title="Training Round",
        yaxis_title="Accuracy",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        yaxis=dict(gridcolor='lightgray')
    )
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Stats
    st.markdown(f"""
    **Statistics:**
    - Initial: {metrics_df.iloc[0]['eval_accuracy']:.4f}
    - Final: {latest['eval_accuracy']:.4f}
    - Improvement: {((latest['eval_accuracy'] - metrics_df.iloc[0]['eval_accuracy']) / metrics_df.iloc[0]['eval_accuracy'] * 100):.2f}%
    - Best: {best_acc_value:.4f} (Round {best_acc_round})
    """)

with tab2:
    fig_prec = go.Figure()
    fig_prec.add_trace(go.Scatter(
        x=metrics_df['round'],
        y=metrics_df['eval_precision'],
        mode='lines+markers',
        name='Precision',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8)
    ))
    fig_prec.update_layout(
        title="Precision Progress Across Rounds",
        xaxis_title="Training Round",
        yaxis_title="Precision",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        yaxis=dict(gridcolor='lightgray')
    )
    st.plotly_chart(fig_prec, use_container_width=True)
    
    # Stats
    best_prec_round = metrics_df['eval_precision'].idxmax() + 1
    best_prec_value = metrics_df['eval_precision'].max()
    st.markdown(f"""
    **Statistics:**
    - Initial: {metrics_df.iloc[0]['eval_precision']:.4f}
    - Final: {latest['eval_precision']:.4f}
    - Improvement: {((latest['eval_precision'] - metrics_df.iloc[0]['eval_precision']) / metrics_df.iloc[0]['eval_precision'] * 100):.2f}%
    - Best: {best_prec_value:.4f} (Round {best_prec_round})
    """)

with tab3:
    fig_rec = go.Figure()
    fig_rec.add_trace(go.Scatter(
        x=metrics_df['round'],
        y=metrics_df['eval_recall'],
        mode='lines+markers',
        name='Recall',
        line=dict(color='#d62728', width=3),
        marker=dict(size=8)
    ))
    fig_rec.update_layout(
        title="Recall Progress Across Rounds",
        xaxis_title="Training Round",
        yaxis_title="Recall",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        yaxis=dict(gridcolor='lightgray')
    )
    st.plotly_chart(fig_rec, use_container_width=True)
    
    # Stats
    best_rec_round = metrics_df['eval_recall'].idxmax() + 1
    best_rec_value = metrics_df['eval_recall'].max()
    st.markdown(f"""
    **Statistics:**
    - Initial: {metrics_df.iloc[0]['eval_recall']:.4f}
    - Final: {latest['eval_recall']:.4f}
    - Improvement: {((latest['eval_recall'] - metrics_df.iloc[0]['eval_recall']) / metrics_df.iloc[0]['eval_recall'] * 100):.2f}%
    - Best: {best_rec_value:.4f} (Round {best_rec_round})
    """)

with tab4:
    fig_f1 = go.Figure()
    fig_f1.add_trace(go.Scatter(
        x=metrics_df['round'],
        y=metrics_df['eval_f1'],
        mode='lines+markers',
        name='F1 Score',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    ))
    fig_f1.update_layout(
        title="F1 Score Progress Across Rounds",
        xaxis_title="Training Round",
        yaxis_title="F1 Score",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        yaxis=dict(gridcolor='lightgray')
    )
    st.plotly_chart(fig_f1, use_container_width=True)
    
    # Stats
    st.markdown(f"""
    **Statistics:**
    - Initial: {metrics_df.iloc[0]['eval_f1']:.4f}
    - Final: {latest['eval_f1']:.4f}
    - Improvement: {((latest['eval_f1'] - metrics_df.iloc[0]['eval_f1']) / metrics_df.iloc[0]['eval_f1'] * 100):.2f}%
    - Best: {best_f1_value:.4f} (Round {best_f1_round})
    """)

with tab5:
    fig_auc = go.Figure()
    fig_auc.add_trace(go.Scatter(
        x=metrics_df['round'],
        y=metrics_df['eval_roc_auc'],
        mode='lines+markers',
        name='ROC AUC',
        line=dict(color='#9467bd', width=3),
        marker=dict(size=8)
    ))
    fig_auc.update_layout(
        title="ROC AUC Progress Across Rounds",
        xaxis_title="Training Round",
        yaxis_title="ROC AUC",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        yaxis=dict(gridcolor='lightgray')
    )
    st.plotly_chart(fig_auc, use_container_width=True)
    
    # Stats
    st.markdown(f"""
    **Statistics:**
    - Initial: {metrics_df.iloc[0]['eval_roc_auc']:.4f}
    - Final: {latest['eval_roc_auc']:.4f}
    - Improvement: {((latest['eval_roc_auc'] - metrics_df.iloc[0]['eval_roc_auc']) / metrics_df.iloc[0]['eval_roc_auc'] * 100):.2f}%
    - Best: {best_auc_value:.4f} (Round {best_auc_round})
    """)

st.markdown("---")

# ===================== COMBINED OVERVIEW PLOT =====================
st.markdown("## 📊 Combined Metrics Overview")

fig_combined = go.Figure()

fig_combined.add_trace(go.Scatter(
    x=metrics_df['round'],
    y=metrics_df['eval_accuracy'],
    mode='lines+markers',
    name='Accuracy',
    line=dict(width=2)
))

fig_combined.add_trace(go.Scatter(
    x=metrics_df['round'],
    y=metrics_df['eval_precision'],
    mode='lines+markers',
    name='Precision',
    line=dict(width=2)
))

fig_combined.add_trace(go.Scatter(
    x=metrics_df['round'],
    y=metrics_df['eval_recall'],
    mode='lines+markers',
    name='Recall',
    line=dict(width=2)
))

fig_combined.add_trace(go.Scatter(
    x=metrics_df['round'],
    y=metrics_df['eval_f1'],
    mode='lines+markers',
    name='F1 Score',
    line=dict(width=2)
))

fig_combined.add_trace(go.Scatter(
    x=metrics_df['round'],
    y=metrics_df['eval_roc_auc'],
    mode='lines+markers',
    name='ROC AUC',
    line=dict(width=2)
))

fig_combined.update_layout(
    title="All Metrics Comparison",
    xaxis_title="Training Round",
    yaxis_title="Score",
    height=600,
    hovermode='x unified',
    plot_bgcolor='white',
    yaxis=dict(gridcolor='lightgray'),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig_combined, use_container_width=True)

st.markdown("---")

# ===================== DATA TABLE =====================
st.markdown("## 📋 Detailed Metrics Table")

# Format the dataframe for display
display_df = metrics_df.copy()
display_df = display_df.round(4)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

# ===================== DOWNLOAD OPTIONS =====================
st.markdown("---")
st.markdown("## 💾 Export Data")

col1, col2, col3 = st.columns(3)

with col1:
    csv_data = metrics_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv_data,
        file_name=f"training_metrics_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

with col2:
    if st.button("🔗 Open MLflow UI"):
        st.markdown(f"[Click here to open MLflow]({MLFLOW_URL})")

with col3:
    if st.button("🔄 Refresh Metrics"):
        st.cache_data.clear()
        st.rerun()

# ===================== FOOTER =====================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>📊 Training Metrics Dashboard | Federated Learning Performance Tracking</p>
    </div>
    """,
    unsafe_allow_html=True
)