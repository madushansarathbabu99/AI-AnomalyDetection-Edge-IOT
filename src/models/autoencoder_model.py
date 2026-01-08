# type: ignore
"""
Autoencoder model for anomaly detection using reconstruction error.
"""

import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from utils.config import (
    AE_MODEL_PATH, AE_THRESHOLD_PATH, AE_ENCODING_DIM,
    AE_EPOCHS, AE_BATCH_SIZE, AE_VALIDATION_SPLIT, RANDOM_SEED
)

# Set seeds for reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class AutoencoderDetector:
    """Autoencoder-based anomaly detection."""
    
    def __init__(self, input_dim=6):
        self.input_dim = input_dim
        self.model = None
        self.threshold = None
        self.scaler = None
    
    def build_model(self):
        """Build autoencoder architecture."""
        # Encoder
        encoder_input = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(4, activation='relu')(encoder_input)
        encoded = layers.Dense(AE_ENCODING_DIM, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(4, activation='relu')(encoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        self.model = models.Model(encoder_input, decoded)
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train, scaler):
        """
        Train autoencoder on normal data only.
        
        Args:
            X_train: Training data (should be mostly normal)
            scaler: Fitted StandardScaler
        """
        print("[AUTOENCODER] Building model...")
        self.scaler = scaler
        self.build_model()
        
        print("[AUTOENCODER] Training...")
        history = self.model.fit(
            X_train, X_train,  # Autoencoder reconstructs input
            epochs=AE_EPOCHS,
            batch_size=AE_BATCH_SIZE,
            validation_split=AE_VALIDATION_SPLIT,
            verbose=0,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Calculate reconstruction errors on training data
        train_reconstructions = self.model.predict(X_train, verbose=0)
        train_errors = np.mean(np.square(X_train - train_reconstructions), axis=1)
        
        # Set threshold at 95th percentile of training errors
        self.threshold = np.percentile(train_errors, 95)
        
        print(f"[AUTOENCODER] Training complete")
        print(f"[AUTOENCODER] Anomaly threshold: {self.threshold:.6f}")
    
    def predict(self, X):
        """
        Predict anomalies based on reconstruction error.
        
        Returns:
            predictions: 0 for normal, 1 for anomaly
        """
        reconstructions = self.model.predict(X, verbose=0)
        errors = np.mean(np.square(X - reconstructions), axis=1)
        return (errors > self.threshold).astype(int)
    
    def predict_scores(self, X):
        """Get reconstruction errors as anomaly scores."""
        reconstructions = self.model.predict(X, verbose=0)
        return np.mean(np.square(X - reconstructions), axis=1)
    
    def save(self, model_path=AE_MODEL_PATH, threshold_path=AE_THRESHOLD_PATH):
        """Save model and threshold."""
        model_path = model_path.replace(".h5", ".keras")
        self.model.save(model_path)  # saves in new format
        joblib.dump(self.threshold, threshold_path)
        print(f"[AUTOENCODER] Model saved: {model_path}")
        print(f"[AUTOENCODER] Threshold saved: {threshold_path}")
    
    
    def load(self, model_path=AE_MODEL_PATH, threshold_path=AE_THRESHOLD_PATH):
        """Load model and threshold."""

    # Ensure we load the modern Keras format if config still points to .h5
        model_path = model_path.replace(".h5", ".keras")

    # compile=False avoids legacy deserialization issues (e.g., keras.metrics.mse)
        self.model = models.load_model(model_path, compile=False)

        self.threshold = joblib.load(threshold_path)

        print(f"[AUTOENCODER] Model loaded: {model_path}")
        print(f"[AUTOENCODER] Threshold loaded: {threshold_path}")
