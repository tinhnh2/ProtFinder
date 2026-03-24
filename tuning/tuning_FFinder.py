#!/usr/bin/env python3
"""
Testing script for FFinder model.

Evaluates a trained FFinder model on test data and generates detailed metrics.

Usage:
    python testing/test_FFinder.py --model_path models/FFinder/FFinder_model.joblib --h5_paths path/to/test.h5
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

sys.path.insert(0, str(Path(__file__).parent.parent))

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


def tuning_FFinder(config, retuning = False):
    """
    Train FFinder model using Balanced Random Forest Classifier.

    Args:
        config: Configuration dictionary
        enable_grid_search: Whether to perform grid search for hyperparameter tuning
    """

    # Load train/val data from the same HDF5 file (different groups)
    print("Loading training and validation data")
    train_val_h5_paths = config['data']['ffinder_joint_h5_paths']
    tuning_h5_paths = config['data']['ffinder_tuning_h5_paths']
    if not train_val_h5_paths or not tuning_h5_paths:
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
    load_dir = Path(config['model_saving']['save_dir'])

    load_path = load_dir / config['model_saving']['filename']
    load_joint_path = load_dir / config['model_saving']['joint_filename']
    pretrained = joblib.load(load_path)
    # ------------------------------------
    # CV + scorer
    # ------------------------------------
    # cv = StratifiedKFold(
    #     n_splits=5,
    #     shuffle=True,
    #     random_state=42
    # )
    #
    # scorer = make_scorer(
    #     roc_auc_score,
    #     needs_proba=True
    # )
    # -----------------------------
    # XGBoost (model-aware)
    # -----------------------------
    print("Training with joint simulation + real data")
    base_clf = XGBClassifier(
        n_estimators=config['training']['n_estimators'],
        max_depth=config['training']['max_depth'],
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        enable_categorical=True,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight
    )
    """
    base_clf = BalancedRandomForestClassifier(
        n_estimators=config['training']['n_estimators'],
        max_depth=config['training']['max_depth'],
        random_state=config['training']['random_state'],
        n_jobs=config['training']['n_jobs']
    )
    """
    # Grid search or direct training
    grid_search = None
    if config['training']['grid_search']['enabled']:
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
        grid_search.fit(X_train, y_train, xgb_model=pretrained.get_booster())

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
        clf.fit(X_train, y_train, xgb_model=pretrained.get_booster())

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

        model_path = save_dir / config['model_saving']['joint_filename']
        joblib.dump(clf, model_path)

        if grid_search is not None:
            print(f"\nModel saved to: {model_path}")
            print("(Best hyperparameters from grid search)")
        else:
            print(f"\nModel saved to: {model_path}")
            print("(Trained with fixed hyperparameters from config)")

    joint_pretrained = joblib.load(load_joint_path)
    # ------------------------------------
    # CV + scorer
    # ------------------------------------
    # cv = StratifiedKFold(
    #     n_splits=5,
    #     shuffle=True,
    #     random_state=42
    # )
    #
    # scorer = make_scorer(
    #     roc_auc_score,
    #     needs_proba=True
    # )
    # -----------------------------
    # XGBoost (model-aware)
    # -----------------------------
    print("Training with joint simulation + real data")
    base_clf = XGBClassifier(
        n_estimators=config['training']['n_estimators'],
        max_depth=config['training']['max_depth'],
        learning_rate=0.001,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        enable_categorical=True,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight
    )
    # Grid search or direct training
    grid_search = None
    if config['training']['grid_search']['enabled']:
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
        grid_search.fit(X_train, y_train, xgb_model=pretrained.get_booster())

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
        clf.fit(X_train, y_train, xgb_model=pretrained.get_booster())

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

        model_path = save_dir / config['model_saving']['tuning_filename']
        joblib.dump(clf, model_path)

        if grid_search is not None:
            print(f"\nModel saved to: {model_path}")
            print("(Best hyperparameters from grid search)")
        else:
            print(f"\nModel saved to: {model_path}")
            print("(Trained with fixed hyperparameters from config)")


def main():
    parser = argparse.ArgumentParser(description="Test FFinder model")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()
    # Load configuration
    config = load_config(args.config)

    tuning_FFinder(config)


if __name__ == "__main__":
    main()
