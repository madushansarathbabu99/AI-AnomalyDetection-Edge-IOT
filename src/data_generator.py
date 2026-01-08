"""
Generate synthetic IoT device behavior data with temporal patterns.
"""

import numpy as np
import pandas as pd
from utils.config import (
    RANDOM_SEED, N_NORMAL, N_ANOMALY, DATA_PATH, FEATURE_COLUMNS
)

np.random.seed(RANDOM_SEED)


def generate_temporal_pattern(n_samples, base_value, amplitude, period):
    """Generate cyclic temporal pattern (e.g., daily cycles)."""
    t = np.arange(n_samples)
    return base_value + amplitude * np.sin(2 * np.pi * t / period)


def generate_synthetic_iot_data(save: bool = True) -> pd.DataFrame:
    """
    Generate synthetic IoT device behavior data with temporal patterns.
    
    Features:
        - cpu_usage (%): 0-100
        - memory_usage (%): 0-100
        - net_in (KB/s): network input
        - net_out (KB/s): network output
        - temperature (°C): device temperature
        - failed_auth (count): failed authentication attempts
    
    Label:
        - 0 = normal
        - 1 = anomaly
    """
    
    # ----- Normal behavior with temporal patterns -----
    # Add daily cycles to make data more realistic
    cpu_base = generate_temporal_pattern(N_NORMAL, 35, 8, 500)
    mem_base = generate_temporal_pattern(N_NORMAL, 40, 10, 500)
    
    normal = pd.DataFrame({
        "cpu_usage": np.clip(
            cpu_base + np.random.normal(0, 8, N_NORMAL), 0, 100
        ),
        "memory_usage": np.clip(
            mem_base + np.random.normal(0, 10, N_NORMAL), 0, 100
        ),
        "net_in": np.random.normal(loc=200, scale=60, size=N_NORMAL).clip(0, None),
        "net_out": np.random.normal(loc=180, scale=50, size=N_NORMAL).clip(0, None),
        "temperature": np.random.normal(loc=45, scale=5, size=N_NORMAL),
        "failed_auth": np.random.poisson(lam=1, size=N_NORMAL),
        "label": 0
    })
    
    # ----- Anomalous behavior patterns -----
    # Create diverse anomaly types
    n_each = N_ANOMALY // 4
    
    # Type 1: Resource exhaustion attack
    resource_attack = pd.DataFrame({
        "cpu_usage": np.random.normal(loc=88, scale=8, size=n_each).clip(0, 100),
        "memory_usage": np.random.normal(loc=92, scale=6, size=n_each).clip(0, 100),
        "net_in": np.random.normal(loc=700, scale=120, size=n_each).clip(0, None),
        "net_out": np.random.normal(loc=650, scale=140, size=n_each).clip(0, None),
        "temperature": np.random.normal(loc=75, scale=7, size=n_each),
        "failed_auth": np.random.poisson(lam=8, size=n_each),
        "label": 1
    })
    
    # Type 2: Network flood
    network_flood = pd.DataFrame({
        "cpu_usage": np.random.normal(loc=65, scale=10, size=n_each).clip(0, 100),
        "memory_usage": np.random.normal(loc=60, scale=12, size=n_each).clip(0, 100),
        "net_in": np.random.normal(loc=1200, scale=200, size=n_each).clip(0, None),
        "net_out": np.random.normal(loc=1100, scale=180, size=n_each).clip(0, None),
        "temperature": np.random.normal(loc=58, scale=6, size=n_each),
        "failed_auth": np.random.poisson(lam=2, size=n_each),
        "label": 1
    })
    
    # Type 3: Brute force authentication
    auth_attack = pd.DataFrame({
        "cpu_usage": np.random.normal(loc=45, scale=12, size=n_each).clip(0, 100),
        "memory_usage": np.random.normal(loc=48, scale=10, size=n_each).clip(0, 100),
        "net_in": np.random.normal(loc=350, scale=80, size=n_each).clip(0, None),
        "net_out": np.random.normal(loc=320, scale=70, size=n_each).clip(0, None),
        "temperature": np.random.normal(loc=52, scale=5, size=n_each),
        "failed_auth": np.random.poisson(lam=25, size=n_each),
        "label": 1
    })
    
    # Type 4: Hardware malfunction (overheating)
    hw_malfunction = pd.DataFrame({
        "cpu_usage": np.random.normal(loc=78, scale=10, size=N_ANOMALY - 3*n_each).clip(0, 100),
        "memory_usage": np.random.normal(loc=82, scale=8, size=N_ANOMALY - 3*n_each).clip(0, 100),
        "net_in": np.random.normal(loc=250, scale=70, size=N_ANOMALY - 3*n_each).clip(0, None),
        "net_out": np.random.normal(loc=240, scale=65, size=N_ANOMALY - 3*n_each).clip(0, None),
        "temperature": np.random.normal(loc=88, scale=8, size=N_ANOMALY - 3*n_each),
        "failed_auth": np.random.poisson(lam=1, size=N_ANOMALY - 3*n_each),
        "label": 1
    })
    
    # Combine all data
    anomaly = pd.concat([
        resource_attack, network_flood, auth_attack, hw_malfunction
    ], ignore_index=True)
    
    df = pd.concat([normal, anomaly], ignore_index=True)
    
    # Add timestamp column for realism
    df['timestamp'] = pd.date_range(
        start='2024-01-01', periods=len(df), freq='1min'
    )
    
    # Shuffle while maintaining some temporal locality
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    if save:
        df.to_csv(DATA_PATH, index=False)
        print(f"[DATA] Synthetic IoT dataset saved: {DATA_PATH}")
        print(f"       Normal samples: {N_NORMAL}, Anomalies: {N_ANOMALY}")
    
    return df
