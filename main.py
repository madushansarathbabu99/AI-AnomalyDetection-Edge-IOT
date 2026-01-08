"""
Main entry point for the Edge IoT Anomaly Detection System.
"""

import os
from src.data_generator import generate_synthetic_iot_data
from src.train_all_models import train_all_models
from src.evaluate_all_models import evaluate_all_models
from src.real_time_simulation import simulate_realtime_detection
from utils.config import DATA_PATH


def main():
    """Run the complete anomaly detection pipeline."""
    
    print("\n" + "="*80)
    print("EDGE IOT ANOMALY DETECTION SYSTEM")
    print("Complete ML Pipeline for IoT Security")
    print("="*80)
    
    # Step 1: Generate data
    if not os.path.exists(DATA_PATH):
        print("\n[STEP 1] Generating synthetic dataset...")
        generate_synthetic_iot_data(save=True)
    else:
        print("\n[STEP 1] Dataset already exists, skipping generation")
    
    # Step 2: Train all models
    print("\n[STEP 2] Training all models...")
    train_all_models()
    
    # Step 3: Evaluate models
    print("\n[STEP 3] Evaluating models...")
    evaluate_all_models()
    
    # Step 4: Real-time simulation
    print("\n[STEP 4] Running real-time detection simulation...")
    simulate_realtime_detection()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review model comparison results in results/model_comparison.json")
    print("  2. Launch dashboard: streamlit run src/dashboard.py")
    print("  3. Integrate with real system data collectors")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()