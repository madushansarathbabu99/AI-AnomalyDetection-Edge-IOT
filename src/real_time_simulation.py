"""
Real-time anomaly detection simulation with all models.
"""

import time
import numpy as np
import joblib
from src.models.isolation_forest_model import IsolationForestDetector
from src.models.autoencoder_model import AutoencoderDetector
from src.models.lstm_model import LSTMDetector
from utils.config import (
    SCALER_PATH, STREAM_SAMPLE_COUNT, STREAM_DELAY_SECONDS, 
    RANDOM_SEED, SEQUENCE_LENGTH
)

np.random.seed(RANDOM_SEED)


def generate_random_sample(anomaly=False):
    """Generate a single random sample."""
    if anomaly:
        # Anomalous sample
        cpu = np.random.normal(85, 10)
        mem = np.random.normal(90, 8)
        net_in = np.random.normal(750, 120)
        net_out = np.random.normal(700, 140)
        temp = np.random.normal(75, 7)
        failed_auth = np.random.poisson(12)
    else:
        # Normal sample
        cpu = np.random.normal(38, 8)
        mem = np.random.normal(42, 10)
        net_in = np.random.normal(220, 50)
        net_out = np.random.normal(200, 45)
        temp = np.random.normal(46, 4)
        failed_auth = np.random.poisson(1)
    
    return np.array([[cpu, mem, net_in, net_out, temp, failed_auth]])


def simulate_realtime_detection():
    """Simulate real-time detection with all three models."""
    
    print("\n" + "="*80)
    print("REAL-TIME ANOMALY DETECTION SIMULATION")
    print("="*80)
    
    # Load models
    print("\n[REALTIME] Loading models...")
    scaler = joblib.load(SCALER_PATH)
    
    iso_model = IsolationForestDetector()
    iso_model.load()
    
    ae_model = AutoencoderDetector()
    ae_model.load()
    
    lstm_model = LSTMDetector()
    lstm_model.load()
    
    # Initialize sequence buffer for LSTM
    sequence_buffer = []
    
    print("\n[REALTIME] Starting simulation...")
    print(f"{'Sample':<8} {'Type':<12} {'ISO-Forest':<15} {'Autoencoder':<15} {'LSTM':<15}")
    print("-" * 80)
    
    for i in range(1, STREAM_SAMPLE_COUNT + 1):
        # Generate sample (inject anomaly every 7th sample)
        is_anomaly = (i % 7 == 0)
        sample = generate_random_sample(anomaly=is_anomaly)
        sample_scaled = scaler.transform(sample)
        
        # Isolation Forest prediction
        pred_iso = iso_model.predict(sample_scaled)[0]
        status_iso = "🔴 ANOMALY" if pred_iso == 1 else "🟢 Normal"
        
        # Autoencoder prediction
        pred_ae = ae_model.predict(sample_scaled)[0]
        status_ae = "🔴 ANOMALY" if pred_ae == 1 else "🟢 Normal"
        
        # LSTM prediction (needs sequence)
        sequence_buffer.append(sample_scaled[0])
        if len(sequence_buffer) > SEQUENCE_LENGTH:
            sequence_buffer.pop(0)
        
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            seq = np.array([sequence_buffer])
            pred_lstm = lstm_model.predict(seq)[0]
            status_lstm = "🔴 ANOMALY" if pred_lstm == 1 else "🟢 Normal"
        else:
            status_lstm = "⏳ Buffering"
        
        # Display
        sample_type = "⚠️  INJECTED" if is_anomaly else "✅ Normal"
        print(f"{i:<8} {sample_type:<12} {status_iso:<15} {status_ae:<15} {status_lstm:<15}")
        
        time.sleep(STREAM_DELAY_SECONDS)
    
    print("\n[REALTIME] Simulation complete")