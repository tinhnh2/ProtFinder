#!/usr/bin/env python3
"""
Training script for FFinder model using Balanced Random Forest Classifier.

Supports grid search for hyperparameter tuning and model saving.

Usage:
    python training/scripts/train_FFinder.py --config configs/FFinder_config.yaml
    python training/scripts/train_FFinder.py --config configs/FFinder_config.yaml --grid_search
"""

import argparse
import yaml
from pathlib import Path
import numpy as np
import h5py
import joblib
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
    
    for h5_path in h5_paths:
        with h5py.File(h5_path, 'r') as h5_file:
            group = h5_file[group_name]
            keys = sorted(group.keys())
            
            for key in keys:
                data = group[key][:].astype(np.float32)
                features.append(data)
                labels.append(int(key[0]))
    
    X = np.vstack(features)
    y = np.array(labels)
    
    return X, y

# -----------------------------
# Utils
# -----------------------------
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

def train_ffinder(config,enable_grid_search=False):
    """
    Train FFinder model using Balanced Random Forest Classifier.
    
    Args:
        config: Configuration dictionary
        enable_grid_search: Whether to perform grid search for hyperparameter tuning
    """

    # Load train/val data from the same HDF5 file (different groups)
    print("Loading training and validation data")
    train_val_h5_paths = config['data']['ffinder_train_val_h5_paths']
    if not train_val_h5_paths:
        raise ValueError("No train/val HDF5 paths specified in config")
    
    X_train, y_train = load_data_from_hdf5(train_val_h5_paths, "train")
    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    
    X_val, y_val = load_data_from_hdf5(train_val_h5_paths, "val")
    print(f"Validation samples: {X_val.shape[0]}, Features: {X_val.shape[1]}")
    
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)

    scale_pos_weight = n_neg / n_pos
    print("scale_pos_weight =", scale_pos_weight)
    # Create classifier
    base_clf2 = BalancedRandomForestClassifier(
        n_estimators=config['training']['n_estimators'],
        max_depth=config['training']['max_depth'],
        random_state=config['training']['random_state'],
        n_jobs=config['training']['n_jobs']
    )
    # -----------------------------
    # XGBoost (model-aware)
    # -----------------------------
    base_clf = XGBClassifier(
        n_estimators=config['training']['n_estimators'],
        max_depth=config['training']['max_depth'],
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc", #logloss
        enable_categorical=True,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight
    )
    # Grid search or direct training
    grid_search = None
    if enable_grid_search or config['training']['grid_search']['enabled']:
        print("\nGrid Search for Hyperparameter Tuning")
        
        grid_config = config['training']['grid_search']
        param_grid = grid_config['param_grid']
        
        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"{param}: {values}")
        
        grid_search = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            cv=grid_config['cv'],
            scoring=grid_config['scoring'],
            n_jobs=config['training']['n_jobs'],
        )
        
        print("Fitting grid search on training data")
        grid_search.fit(X_train, y_train)
        
        print("\nGrid Search Results")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score ({grid_config['scoring']}): {grid_search.best_score_:.4f}")
        
        # Use best model
        clf = grid_search.best_estimator_
        
        # Evaluate on validation set
        print("\nValidation Set Performance using Best Model")
        y_val_pred = clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
        
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"Balanced Accuracy: {val_balanced_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_val_pred, target_names=['-F', '+F'], digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_val_pred))
        
    else:
        print("\nTraining Model (No Grid Search)")
        
        # Train model
        print("Fitting model on training data")
        clf = base_clf
        clf.fit(X_train, y_train)
        
        # Evaluate on validation set
        print("\nValidation Set Performance")
        y_val_pred = clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
        
        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"Balanced Accuracy: {val_balanced_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_val_pred, target_names=['-F', '+F'], digits=4))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_val_pred))
    
    # Save model
    if config['model_saving']['save_model']:
        save_dir = Path(config['model_saving']['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / config['model_saving']['filename']
        joblib.dump(clf, model_path)
        
        if grid_search is not None:
            print(f"\nModel saved to: {model_path}")
            print("(Best hyperparameters from grid search)")
        else:
            print(f"\nModel saved to: {model_path}")
            print("(Trained with fixed hyperparameters from config)")


def main():
    import time
    start_total = time.perf_counter()
    parser = argparse.ArgumentParser(description="Train FFinder model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--grid_search",
        action="store_true",
        help="Enable grid search for hyperparameter tuning (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Train model
    train_ffinder(config, enable_grid_search=args.grid_search)
    end_total = time.perf_counter()
    total_time = end_total - start_total

    print(f"Total execution time: {total_time:.2f} seconds")
    
    print("Training completed")


if __name__ == "__main__":
    main()
