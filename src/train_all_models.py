"""
Train all three anomaly detection models.
"""

import joblib
from sklearn.model_selection import train_test_split
from src.feature_processing import (
    load_dataset, split_features_labels, scale_features,
    create_sequences, save_sequence_data
)
from src.models.isolation_forest_model import IsolationForestDetector
from src.models.autoencoder_model import AutoencoderDetector
from src.models.lstm_model import LSTMDetector
from utils.config import TEST_SIZE, RANDOM_SEED, SCALER_PATH


def train_all_models():
    """Train Isolation Forest, Autoencoder, and LSTM models."""
    
    print("\n" + "="*80)
    print("TRAINING ALL MODELS")
    print("="*80)
    
    # ===== Load and preprocess data =====
    print("\n[TRAIN] Loading dataset...")
    df = load_dataset()
    X, y = split_features_labels(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    
    print(f"[TRAIN] Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"[TRAIN] Train anomalies: {y_train.sum()}, Test anomalies: {y_test.sum()}")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"[TRAIN] Scaler saved: {SCALER_PATH}")
    
    # ===== Train Isolation Forest =====
    print("\n" + "-"*80)
    iso_model = IsolationForestDetector()
    iso_model.train(X_train_scaled, scaler)
    iso_model.save()
    
    # ===== Train Autoencoder =====
    print("\n" + "-"*80)
    # Filter to mostly normal data for autoencoder training
    X_train_normal = X_train_scaled[y_train == 0]
    ae_model = AutoencoderDetector(input_dim=X_train.shape[1])
    ae_model.train(X_train_normal, scaler)
    ae_model.save()
    
    # ===== Prepare sequences for LSTM =====
    print("\n" + "-"*80)
    print("[TRAIN] Creating sequences for LSTM...")
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test)
    
    print(f"[TRAIN] Train sequences: {len(X_train_seq)}, Test sequences: {len(X_test_seq)}")
    
    # Save sequences
    save_sequence_data(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
    
    # ===== Train LSTM =====
    lstm_model = LSTMDetector(
        sequence_length=X_train_seq.shape[1],
        n_features=X_train_seq.shape[2]
    )
    lstm_model.train(X_train_seq, y_train_seq, scaler)
    lstm_model.save()
    
    print("\n" + "="*80)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("="*80)
    
    return {
        'iso_model': iso_model,
        'ae_model': ae_model,
        'lstm_model': lstm_model,
        'scaler': scaler,
        'test_data': {
            'X_test_scaled': X_test_scaled,
            'y_test': y_test,
            'X_test_seq': X_test_seq,
            'y_test_seq': y_test_seq
        }
    }

