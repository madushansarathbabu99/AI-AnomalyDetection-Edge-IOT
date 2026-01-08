"""
Feature preprocessing and sequence generation for temporal models.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler
from utils.config import (
    DATA_PATH, SCALER_PATH, FEATURE_COLUMNS, 
    SEQUENCE_LENGTH, SEQUENCE_STRIDE, SEQUENCES_PATH
)


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset from CSV."""
    df = pd.read_csv(path)
    return df


def split_features_labels(df: pd.DataFrame):
    """Extract features and labels."""
    X = df[FEATURE_COLUMNS].to_numpy()
    y = df["label"].to_numpy().astype(int)
    return X, y


"""def scale_features(X_train, X_test=None):
    
    Fit StandardScaler on training data and transform.
    
    Returns:
        X_train_scaled, X_test_scaled (if provided), scaler
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = None
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler
"""

def scale_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """
    Fit StandardScaler on training data and transform.
    Always returns 3 values:
      (X_train_scaled, X_test_scaled, scaler)
    where X_test_scaled is None if X_test is not provided.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_test_scaled, scaler

def create_sequences(X, y, sequence_length=SEQUENCE_LENGTH, stride=SEQUENCE_STRIDE):
    """
    Create sliding window sequences for LSTM.
    
    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        sequence_length: Number of timesteps per sequence
        stride: Step size for sliding window
    
    Returns:
        X_seq: Sequences of shape (n_sequences, sequence_length, n_features)
        y_seq: Labels of shape (n_sequences,) - label of last timestep
    """
    X_seq, y_seq = [], []
    
    for i in range(0, len(X) - sequence_length + 1, stride):
        X_seq.append(X[i:i + sequence_length])
        # Use label of the last timestep in sequence
        y_seq.append(y[i + sequence_length - 1])
    
    return np.array(X_seq), np.array(y_seq)


def save_sequence_data(X_train_seq, y_train_seq, X_test_seq, y_test_seq):
    """Save sequence data to disk."""
    np.savez(
        SEQUENCES_PATH,
        X_train_seq=X_train_seq,
        y_train_seq=y_train_seq,
        X_test_seq=X_test_seq,
        y_test_seq=y_test_seq
    )
    print(f"[DATA] Sequence data saved: {SEQUENCES_PATH}")


def load_sequence_data():
    """Load pre-generated sequence data."""
    data = np.load(SEQUENCES_PATH)
    return (
        data['X_train_seq'],
        data['y_train_seq'],
        data['X_test_seq'],
        data['y_test_seq']
    )
