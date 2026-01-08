"""
Central configuration for the entire anomaly detection system.
"""

import os

# ===== Random seed for reproducibility =====
RANDOM_SEED = 42

# ===== Dataset configuration =====
N_NORMAL = 8000
N_ANOMALY = 500

# ===== Train-test split ratio =====
TEST_SIZE = 0.3

# ===== LSTM sequence configuration =====
SEQUENCE_LENGTH = 10  # Number of timesteps per sequence
SEQUENCE_STRIDE = 1   # Sliding window stride

# ===== Model hyperparameters =====
# Isolation Forest
ISO_N_ESTIMATORS = 150
ISO_CONTAMINATION = 'auto'

# Autoencoder
AE_ENCODING_DIM = 3
AE_EPOCHS = 50
AE_BATCH_SIZE = 32
AE_VALIDATION_SPLIT = 0.2

# LSTM
LSTM_UNITS = 32
LSTM_EPOCHS = 30
LSTM_BATCH_SIZE = 64

# ===== Directory paths =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create directories
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ===== File paths =====
DATA_PATH = os.path.join(DATA_DIR, "synthetic_iot_data.csv")
SEQUENCES_PATH = os.path.join(DATA_DIR, "sequence_data.npz")

ISO_MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
AE_MODEL_PATH = os.path.join(MODELS_DIR, "autoencoder.h5")
AE_THRESHOLD_PATH = os.path.join(MODELS_DIR, "autoencoder_threshold.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

RESULTS_PATH = os.path.join(RESULTS_DIR, "model_comparison.json")

# ===== Real-time simulation settings =====
STREAM_SAMPLE_COUNT = 50
STREAM_DELAY_SECONDS = 0.5

# ===== Feature columns =====
FEATURE_COLUMNS = [
    "cpu_usage",
    "memory_usage", 
    "net_in",
    "net_out",
    "temperature",
    "failed_auth",
]