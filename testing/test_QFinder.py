#!/usr/bin/env python3
"""
Testing script for QFinder model.

Evaluates a trained QFinder model on test data and generates detailed metrics.

Usage:
    python testing/test_QFinder.py --checkpoint path/to/checkpoint.ckpt --test_h5_paths path/to/test.h5
"""

import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
import pytorch_lightning as pl

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.modules import QFinderLightningModule
from data import QFinderDataset
from testing.callbacks import PredictionCollector


def create_data_loader(test_h5_paths, batch_size):
    """
    Create test data loader.
    
    Args:
        test_h5_paths: List of HDF5 file paths for testing
        batch_size: Batch size for testing
    
    Returns:
        DataLoader instance
    """
    test_dataset = QFinderDataset(h5_paths=test_h5_paths, group_name="test")
    
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Must be 0 for HDF5 files
    )
    
    return test_loader

import numpy as np
def topk_accuracy(y_true, y_prob, k=3):
    topk = np.argsort(y_prob, axis=1)[:, -k:]
    return np.mean([y in topk[i] for i, y in enumerate(y_true)])

def test_QFinder(
    checkpoint_path: str,
    test_h5_paths: list,
    batch_size: int,
    top_k = 3
):
    """
    Test QFinder model on test dataset using PyTorch Lightning.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_h5_paths: List of HDF5 file paths for testing
        batch_size: Batch size for testing
    """

    # Load model from checkpoint
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = QFinderLightningModule.load_from_checkpoint(checkpoint_path)
    
    # Create test data loader
    print("Loading test data")
    test_loader = create_data_loader(test_h5_paths, batch_size)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create prediction collector callback
    prediction_collector = PredictionCollector()
    
    # Create trainer for testing
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        callbacks=prediction_collector
    )
    
    # Run test using Lightning
    print("Running test")
    trainer.test(model, dataloaders=test_loader)
    
    # Get collected predictions
    all_labels, all_predictions, y_probs = prediction_collector.get_results()

    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    
    # Class names
    class_names = ['LG', 'WAG', 'JTT', 'Q.plant', 'Q.bird', 'Q.mammal', 'Q.pfam']
    print("=== Classification report ===")
    print(classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    ))
    """
    # Generate classification report
    report_str = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4
    )
    """
    for k in [1, 2, 3]:
        print(f"Top-{k} accuracy: {topk_accuracy(all_labels, y_probs, k):.4f}")

    print("\n=== Example top-k predictions ===")
    for i in range(5):
        topk = np.argsort(y_probs[i])[-top_k:][::-1]
        print(
            f"True: {class_names[all_labels[i]]} | "
            f"Top-{top_k}: {[class_names[j] for j in topk]}"
        )
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Print results
    print("\nTest Results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print("\nClassification Report:")
    # print(report_str)
    print("\nConfusion Matrix:")
    print(cm)


def main():
    parser = argparse.ArgumentParser(description="Test QFinder model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test_h5_paths",
        type=str,
        nargs="+",
        default=["./hdf5_features/QFinder_feature_test.h5"],
        help="Paths to test HDF5 files (default: ./hdf5_features/QFinder_feature_test.h5)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=True,
        help="Top-K statistics"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for testing"
    )
    
    args = parser.parse_args()
    
    import time
    start_total = time.perf_counter()
    test_QFinder(
        checkpoint_path=args.checkpoint,
        test_h5_paths=args.test_h5_paths,
        batch_size=args.batch_size,
        top_k=args.top_k
    )
    end_total = time.perf_counter()
    total_time = end_total - start_total

    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
