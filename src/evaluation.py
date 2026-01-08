"""
Comprehensive model evaluation and comparison framework.
"""

import time
import json
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_recall_fscore_support
)
from utils.config import RESULTS_PATH


class ModelEvaluator:
    """Evaluate and compare anomaly detection models."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model_name, y_true, y_pred, y_scores=None, 
                       inference_times=None):
        """
        Evaluate a single model comprehensively.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Anomaly scores (optional, for ROC-AUC)
            inference_times: List of inference times per sample
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION: {model_name}")
        print(f"{'='*60}")
        
        # Classification metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # ROC-AUC if scores provided
        roc_auc = None
        if y_scores is not None:
            try:
                roc_auc = roc_auc_score(y_true, y_scores)
            except:
                roc_auc = None
        
        # Inference time statistics
        avg_inference_time = None
        if inference_times is not None:
            avg_inference_time = np.mean(inference_times) * 1000  # ms
        
        # Store results
        self.results[model_name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr),
            'avg_inference_time_ms': float(avg_inference_time) if avg_inference_time else None
        }
        
        # Print detailed report
        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4, 
                                    target_names=['Normal', 'Anomaly']))
        
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                Normal  Anomaly")
        print(f"Actual Normal   {tn:6d}  {fp:6d}")
        print(f"       Anomaly  {fn:6d}  {tp:6d}")
        
        print(f"\nKey Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  FPR:       {fpr:.4f}")
        print(f"  FNR:       {fnr:.4f}")
        if avg_inference_time:
            print(f"  Avg Inference: {avg_inference_time:.2f} ms")
    
    def measure_inference_time(self, model, X, n_runs=100):
        """Measure average inference time."""
        times = []
        
        # Warmup
        for _ in range(10):
            _ = model.predict(X[:10])
        
        # Measure
        for i in range(min(n_runs, len(X))):
            start = time.perf_counter()
            _ = model.predict(X[i:i+1])
            end = time.perf_counter()
            times.append(end - start)
        
        return times
    
    def save_results(self, path=RESULTS_PATH):
        """Save comparison results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n[EVAL] Results saved: {path}")
    
    def print_comparison(self):
        """Print side-by-side model comparison."""
        if not self.results:
            print("No results to compare")
            return
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Header
        print(f"\n{'Metric':<20}", end='')
        for model_name in self.results.keys():
            print(f"{model_name:<20}", end='')
        print()
        print('-' * (20 + 20 * len(self.results)))
        
        # Metrics
        for metric in metrics:
            print(f"{metric.replace('_', ' ').title():<20}", end='')
            for model_name in self.results.keys():
                value = self.results[model_name].get(metric)
                if value is not None:
                    print(f"{value:<20.4f}", end='')
                else:
                    print(f"{'N/A':<20}", end='')
            print()
        
        # Inference time
        print(f"{'Inference (ms)':<20}", end='')
        for model_name in self.results.keys():
            value = self.results[model_name].get('avg_inference_time_ms')
            if value is not None:
                print(f"{value:<20.2f}", end='')
            else:
                print(f"{'N/A':<20}", end='')
        print()
        
        print()
