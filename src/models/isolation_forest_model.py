# type: ignore
"""
Isolation Forest model for anomaly detection.
"""

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from utils.config import (
    ISO_MODEL_PATH, ISO_N_ESTIMATORS, ISO_CONTAMINATION, RANDOM_SEED
)


class IsolationForestDetector:
    """Wrapper for Isolation Forest anomaly detection."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
    
    def train(self, X_train, scaler):
        """Train Isolation Forest model."""
        print("[ISO-FOREST] Training Isolation Forest...")
        
        self.scaler = scaler
        self.model = IsolationForest(
            n_estimators=ISO_N_ESTIMATORS,
            contamination=ISO_CONTAMINATION,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        self.model.fit(X_train)
        
        print(f"[ISO-FOREST] Training complete")
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Returns:
            predictions: 0 for normal, 1 for anomaly
        """
        raw_preds = self.model.predict(X)
        # Convert: 1 (normal) -> 0, -1 (anomaly) -> 1
        return np.where(raw_preds == -1, 1, 0)
    
    def predict_scores(self, X):
        """Get anomaly scores (lower is more anomalous)."""
        return self.model.score_samples(X)
    
    def save(self, path=ISO_MODEL_PATH):
        """Save model to disk."""
        joblib.dump(self.model, path)
        print(f"[ISO-FOREST] Model saved: {path}")
    
    def load(self, path=ISO_MODEL_PATH):
        """Load model from disk."""
        self.model = joblib.load(path)
        print(f"[ISO-FOREST] Model loaded: {path}")