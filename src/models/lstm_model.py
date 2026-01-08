# type: ignore
"""
LSTM model for temporal anomaly detection.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, metrics
from utils.config import (
    LSTM_MODEL_PATH, LSTM_UNITS, LSTM_EPOCHS, 
    LSTM_BATCH_SIZE, SEQUENCE_LENGTH, RANDOM_SEED
)

# Set seeds
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class LSTMDetector:
    """LSTM-based temporal anomaly detection."""
    
    def __init__(self, sequence_length=SEQUENCE_LENGTH, n_features=6):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = None
    
    def build_model(self):
        """Build LSTM classification model."""
        model = models.Sequential([
            layers.LSTM(
                LSTM_UNITS,
                input_shape=(self.sequence_length, self.n_features),
                return_sequences=True
            ),
            layers.Dropout(0.2),
            layers.LSTM(LSTM_UNITS // 2),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', metrics.Precision(), metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def train(self, X_train_seq, y_train_seq, scaler):
        """
        Train LSTM model on sequences.
        
        Args:
            X_train_seq: Training sequences (n_seq, seq_len, n_features)
            y_train_seq: Training labels (n_seq,)
            scaler: Fitted StandardScaler
        """
        print("[LSTM] Building model...")
        self.scaler = scaler
        self.build_model()
        
        print("[LSTM] Training...")
        
        # Handle class imbalance
        pos_weight = np.sum(y_train_seq == 0) / np.sum(y_train_seq == 1)
        class_weight = {0: 1.0, 1: pos_weight}
        
        history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_split=0.2,
            class_weight=class_weight,
            verbose=0,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        print(f"[LSTM] Training complete")
    
    def predict(self, X_seq):
        """
        Predict anomalies from sequences.
        
        Returns:
            predictions: 0 for normal, 1 for anomaly
        """
        probs = self.model.predict(X_seq, verbose=0)
        return (probs.flatten() > 0.5).astype(int)
    
    def predict_proba(self, X_seq):
        """Get anomaly probabilities."""
        return self.model.predict(X_seq, verbose=0).flatten()
    
    def save(self, path=LSTM_MODEL_PATH):
        """Save model."""
        self.model.save(path)
        print(f"[LSTM] Model saved: {path}")
    
    def load(self, path=LSTM_MODEL_PATH):
        """Load model."""
        self.model = models.load_model(path)
        print(f"[LSTM] Model loaded: {path}")
