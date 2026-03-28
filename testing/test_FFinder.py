#!/usr/bin/env python3
"""
Testing script for FFinder model.

Evaluates a trained FFinder model on test data and generates detailed metrics.

Usage:
    python testing/test_FFinder.py --model_path models/FFinder/FFinder_model.joblib --h5_paths path/to/test.h5
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data_from_hdf5(h5_paths, group_name):
    """
    Load features and labels from HDF5 files.
    
    Args:
        h5_paths: List of HDF5 file paths
        group_name: Name of the HDF5 group (e.g., "train", "val", "test")
    
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    if isinstance(h5_paths, str):
        h5_paths = [h5_paths]
    
    features = []
    labels = []
    keys_all = []
    
    for h5_path in h5_paths:
        with h5py.File(h5_path, 'r') as h5_file:
            group = h5_file[group_name]
            keys = sorted(group.keys())
            
            for key in keys:
                data = group[key][:].astype(np.float32)
                features.append(data)
                labels.append(int(key[0]))
                keys_all.append(key)
    
    X = np.vstack(features)
    y = np.array(labels)
    
    return X, y, keys_all

def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_score = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = balanced_accuracy_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score

def test_FFinder(model_path: str, h5_paths: list):
    """
    Test FFinder model on test dataset.
    
    Args:
        model_path: Path to trained model (joblib file)
        h5_paths: List of HDF5 file paths for testing
    """

    print(f"Loading model from: {model_path}")
    clf = joblib.load(model_path)
    
    print("Loading test data")
    X, y, keys = load_data_from_hdf5(h5_paths, "test")
    print(f"Test samples: {X.shape[0]}, Features: {X.shape[1]}")
    
    print("Running predictions")
    y_prob = clf.predict_proba(X)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    accuracy = accuracy_score(y, y_pred)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    class_names = ['-F', '+F']
    # ===== ✅ Convert label index → name =====
    true_names = [class_names[x] for x in y]
    pred_names = [class_names[x] for x in y_pred]

    # ===== ✅ Save CSV =====
    df = pd.DataFrame({
        "alignment": keys,
        "true_label": true_names,
        "predicted_label": pred_names
    })

    df["correct"] = (df["true_label"] == df["predicted_label"])

    df.to_csv("results_ffinder.csv", index=False)

    print("\nSaved results to results_ffinder.csv")
    
    report_str = classification_report(
        y,
        y_pred,
        target_names=class_names,
        digits=4
    )
    
    cm = confusion_matrix(y, y_pred)
    
    print("\nTest Results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\nClassification Report:")
    print(report_str)
    print("\nConfusion Matrix:")
    print(cm)

    # =====================
    # ROC - AUC
    # =====================
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)

    print(f"\n[INFO] ROC-AUC: {roc_auc:.4f}")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (+F vs -F)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("roc_auc_test.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Test FFinder model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model (joblib file)"
    )
    parser.add_argument(
        "--test_h5_paths",
        type=str,
        nargs="+",
        default=["./hdf5_features/FFinder_feature_test.h5"],
        help="Paths to HDF5 files for testing (default: ./hdf5_features/FFinder_feature_test.h5)"
    )
    
    args = parser.parse_args()
    
    import time
    start_total = time.perf_counter()
    test_FFinder(
        model_path=args.checkpoint,
        h5_paths=args.h5_test_paths
    )
    end_total = time.perf_counter()
    total_time = end_total - start_total

    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
