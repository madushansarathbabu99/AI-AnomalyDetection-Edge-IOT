"""
Interactive Streamlit dashboard for model comparison and visualization.

Run with: streamlit run dashboard.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import time
from src.models.isolation_forest_model import IsolationForestDetector
from src.models.autoencoder_model import AutoencoderDetector
from src.models.lstm_model import LSTMDetector
from src.feature_processing import load_dataset, split_features_labels
from utils.config import RESULTS_PATH, SCALER_PATH, FEATURE_COLUMNS, SEQUENCE_LENGTH
import joblib
from sklearn.model_selection import train_test_split
from utils.config import TEST_SIZE, RANDOM_SEED


st.set_page_config(
    page_title="Edge IoT Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_models():
    """Load all trained models."""
    scaler = joblib.load(SCALER_PATH)
    
    iso_model = IsolationForestDetector()
    iso_model.load()
    
    ae_model = AutoencoderDetector()
    ae_model.load()
    
    lstm_model = LSTMDetector()
    lstm_model.load()
    
    return iso_model, ae_model, lstm_model, scaler


@st.cache_data
def load_evaluation_results():
    """Load evaluation results."""
    try:
        with open(RESULTS_PATH, 'r') as f:
            return json.load(f)
    except:
        return None


def main():
    st.title("🔍 Edge IoT Anomaly Detection System")
    st.markdown("### Real-time monitoring and model comparison dashboard")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Model Comparison", "🔴 Live Detection", "📈 Data Exploration"]
    )
    
    if page == "📊 Model Comparison":
        show_model_comparison()
    elif page == "🔴 Live Detection":
        show_live_detection()
    else:
        show_data_exploration()


def show_model_comparison():
    """Display model comparison metrics."""
    st.header("Model Performance Comparison")
    
    results = load_evaluation_results()
    
    if results is None:
        st.warning("No evaluation results found. Please run evaluation first.")
        return
    
    # Metrics overview
    col1, col2, col3 = st.columns(3)
    
    models = list(results.keys())
    
    with col1:
        st.metric(
            "Best F1-Score",
            f"{max(results[m]['f1_score'] for m in models):.3f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Best Precision",
            f"{max(results[m]['precision'] for m in models):.3f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Best Recall",
            f"{max(results[m]['recall'] for m in models):.3f}",
            delta=None
        )
    
    # Comparison table
    st.subheader("Detailed Metrics")
    
    df_results = pd.DataFrame(results).T
    df_results = df_results[['accuracy', 'precision', 'recall', 'f1_score', 
                              'roc_auc', 'false_positive_rate', 
                              'avg_inference_time_ms']]
    
    st.dataframe(df_results.style.highlight_max(axis=0, color='lightgreen'), 
                 use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance metrics comparison
        metrics = ['precision', 'recall', 'f1_score']
        fig = go.Figure()
        
        for model in models:
            fig.add_trace(go.Bar(
                name=model,
                x=metrics,
                y=[results[model][m] for m in metrics]
            ))
        
        fig.update_layout(
            title="Performance Metrics Comparison",
            barmode='group',
            yaxis_title="Score",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confusion matrices
        model_select = st.selectbox("Select Model", models)
        
        tp = results[model_select]['true_positives']
        tn = results[model_select]['true_negatives']
        fp = results[model_select]['false_positives']
        fn = results[model_select]['false_negatives']
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Normal', 'Anomaly'],
            y=['Normal', 'Anomaly'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig.update_layout(title=f"Confusion Matrix - {model_select}")
        st.plotly_chart(fig, use_container_width=True)


def show_live_detection():
    """Show live anomaly detection simulation."""
    st.header("Live Anomaly Detection")
    
    try:
        iso_model, ae_model, lstm_model, scaler = load_models()
    except:
        st.error("Models not found. Please train models first.")
        return
    
    st.markdown("Simulating real-time IoT device monitoring...")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.slider("Number of samples", 10, 100, 30)
    with col2:
        anomaly_freq = st.slider("Anomaly frequency (every N samples)", 5, 20, 7)
    with col3:
        start_button = st.button("Start Simulation", type="primary")
    
    if start_button:
        # Placeholders
        status_placeholder = st.empty()
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Data storage
        history = {
            'sample': [],
            'cpu': [],
            'memory': [],
            'network': [],
            'iso_pred': [],
            'ae_pred': [],
            'lstm_pred': [],
            'is_anomaly': []
        }
        
        sequence_buffer = []
        
        for i in range(1, n_samples + 1):
            # Generate sample
            is_anomaly = (i % anomaly_freq == 0)
            
            if is_anomaly:
                sample = np.array([[
                    np.random.normal(85, 10),
                    np.random.normal(90, 8),
                    np.random.normal(750, 120),
                    np.random.normal(700, 140),
                    np.random.normal(75, 7),
                    np.random.poisson(12)
                ]])
            else:
                sample = np.array([[
                    np.random.normal(38, 8),
                    np.random.normal(42, 10),
                    np.random.normal(220, 50),
                    np.random.normal(200, 45),
                    np.random.normal(46, 4),
                    np.random.poisson(1)
                ]])
            
            sample_scaled = scaler.transform(sample)
            
            # Predictions
            pred_iso = iso_model.predict(sample_scaled)[0]
            pred_ae = ae_model.predict(sample_scaled)[0]
            
            # LSTM (needs sequence)
            sequence_buffer.append(sample_scaled[0])
            if len(sequence_buffer) > SEQUENCE_LENGTH:
                sequence_buffer.pop(0)
            
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                seq = np.array([sequence_buffer])
                pred_lstm = lstm_model.predict(seq)[0]
            else:
                pred_lstm = -1  # Not ready
            
            # Store history
            history['sample'].append(i)
            history['cpu'].append(sample[0][0])
            history['memory'].append(sample[0][1])
            history['network'].append(sample[0][2])
            history['iso_pred'].append(pred_iso)
            history['ae_pred'].append(pred_ae)
            history['lstm_pred'].append(pred_lstm if pred_lstm != -1 else 0)
            history['is_anomaly'].append(1 if is_anomaly else 0)
            
            # Update status
            status_text = f"**Sample {i}/{n_samples}**"
            if is_anomaly:
                status_text += " - ⚠️ **INJECTED ANOMALY**"
            
            status_text += f"\n\n"
            status_text += f"- Isolation Forest: {'🔴 ANOMALY' if pred_iso == 1 else '🟢 Normal'}\n"
            status_text += f"- Autoencoder: {'🔴 ANOMALY' if pred_ae == 1 else '🟢 Normal'}\n"
            status_text += f"- LSTM: {'🔴 ANOMALY' if pred_lstm == 1 else '🟢 Normal' if pred_lstm == 0 else '⏳ Buffering'}\n"
            
            status_placeholder.markdown(status_text)
            
            # Update chart
            df_history = pd.DataFrame(history)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_history['sample'],
                y=df_history['cpu'],
                name='CPU Usage',
                line=dict(color='blue')
            ))
            
            # Mark anomalies
            anomaly_samples = df_history[df_history['is_anomaly'] == 1]['sample']
            anomaly_cpu = df_history[df_history['is_anomaly'] == 1]['cpu']
            fig.add_trace(go.Scatter(
                x=anomaly_samples,
                y=anomaly_cpu,
                mode='markers',
                name='Injected Anomaly',
                marker=dict(color='red', size=12, symbol='x')
            ))
            
            fig.update_layout(
                title="Real-time CPU Usage Monitoring",
                xaxis_title="Sample",
                yaxis_title="CPU Usage (%)",
                height=400
            )
            
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Update metrics
            if len(history['sample']) > 1:
                col1, col2, col3 = metrics_placeholder.columns(3)
                
                iso_correct = sum(1 for j in range(len(history['is_anomaly'])) 
                                 if history['iso_pred'][j] == history['is_anomaly'][j])
                ae_correct = sum(1 for j in range(len(history['is_anomaly'])) 
                                if history['ae_pred'][j] == history['is_anomaly'][j])
                lstm_correct = sum(1 for j in range(len(history['is_anomaly'])) 
                                  if history['lstm_pred'][j] == history['is_anomaly'][j])
                
                with col1:
                    st.metric("Isolation Forest Accuracy", 
                             f"{iso_correct / len(history['is_anomaly']):.2%}")
                with col2:
                    st.metric("Autoencoder Accuracy", 
                             f"{ae_correct / len(history['is_anomaly']):.2%}")
                with col3:
                    st.metric("LSTM Accuracy", 
                             f"{lstm_correct / len(history['is_anomaly']):.2%}")
            
            time.sleep(0.5)
        
        st.success("Simulation complete!")


def show_data_exploration():
    """Show dataset exploration and visualization."""
    st.header("Dataset Exploration")
    
    df = load_dataset()
    
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Normal Samples", (df['label'] == 0).sum())
    with col3:
        st.metric("Anomalies", (df['label'] == 1).sum())
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    feature = st.selectbox("Select Feature", FEATURE_COLUMNS)
    
    fig = go.Figure()
    
    normal_data = df[df['label'] == 0][feature]
    anomaly_data = df[df['label'] == 1][feature]
    
    fig.add_trace(go.Histogram(
        x=normal_data,
        name='Normal',
        opacity=0.7,
        marker_color='green'
    ))
    
    fig.add_trace(go.Histogram(
        x=anomaly_data,
        name='Anomaly',
        opacity=0.7,
        marker_color='red'
    ))
    
    fig.update_layout(
        title=f"{feature.replace('_', ' ').title()} Distribution",
        xaxis_title=feature.replace('_', ' ').title(),
        yaxis_title="Count",
        barmode='overlay',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    
    corr_matrix = df[FEATURE_COLUMNS].corr()
    
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=FEATURE_COLUMNS,
        y=FEATURE_COLUMNS,
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    fig.update_layout(height=600)
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()