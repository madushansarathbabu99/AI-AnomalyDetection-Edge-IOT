"""
Evaluate and compare all trained models.
"""

import numpy as np
from src.evaluation import ModelEvaluator
from src.models.isolation_forest_model import IsolationForestDetector
from src.models.autoencoder_model import AutoencoderDetector
from src.models.lstm_model import LSTMDetector
from src.feature_processing import load_sequence_data
from sklearn.model_selection import train_test_split
from src.feature_processing import load_dataset, split_features_labels, scale_features
from utils.config import TEST_SIZE, RANDOM_SEED
import joblib
from utils.config import SCALER_PATH


def evaluate_all_models():
    """Load and evaluate all trained models."""
    
    print("\n" + "="*80)
    print("EVALUATING ALL MODELS")
    print("="*80)
    
    # Load test data
    print("\n[EVAL] Loading test data...")
    df = load_dataset()
    X, y = split_features_labels(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    
    scaler = joblib.load(SCALER_PATH)
    X_test_scaled = scaler.transform(X_test)
    
    # Load sequence data
    X_train_seq, y_train_seq, X_test_seq, y_test_seq = load_sequence_data()
    
    evaluator = ModelEvaluator()
    
    # ===== Evaluate Isolation Forest =====
    print("\n[EVAL] Loading Isolation Forest...")
    iso_model = IsolationForestDetector()
    iso_model.load()
    iso_model.scaler = scaler
    
    y_pred_iso = iso_model.predict(X_test_scaled)
    y_scores_iso = -iso_model.predict_scores(X_test_scaled)  # Negate for consistency
    times_iso = evaluator.measure_inference_time(iso_model, X_test_scaled)
    
    evaluator.evaluate_model(
        'Isolation Forest',
        y_test, y_pred_iso, y_scores_iso, times_iso
    )
    
    # ===== Evaluate Autoencoder =====
    print("\n[EVAL] Loading Autoencoder...")
    ae_model = AutoencoderDetector()
    ae_model.load()
    ae_model.scaler = scaler
    
    y_pred_ae = ae_model.predict(X_test_scaled)
    y_scores_ae = ae_model.predict_scores(X_test_scaled)
    times_ae = evaluator.measure_inference_time(ae_model, X_test_scaled)
    
    evaluator.evaluate_model(
        'Autoencoder',
        y_test, y_pred_ae, y_scores_ae, times_ae
    )
    
    # ===== Evaluate LSTM =====
    print("\n[EVAL] Loading LSTM...")
    lstm_model = LSTMDetector()
    lstm_model.load()
    lstm_model.scaler = scaler
    
    y_pred_lstm = lstm_model.predict(X_test_seq)
    y_scores_lstm = lstm_model.predict_proba(X_test_seq)
    times_lstm = evaluator.measure_inference_time(lstm_model, X_test_seq)
    
    evaluator.evaluate_model(
        'LSTM',
        y_test_seq, y_pred_lstm, y_scores_lstm, times_lstm
    )
    
    # ===== Print comparison =====
    evaluator.print_comparison()
    evaluator.save_results()
    
    return evaluator
