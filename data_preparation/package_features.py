#!/usr/bin/env python3
"""
Package extracted features (.npy files) into HDF5 format.

This script splits features from feature_extraction.py into train/val/test sets
and packages them into HDF5 files for efficient loading during training.

Initial reason for packaging to HDF5 format: The supercomputer Gadi has a limit on the number of files that can be stored.
Besides, HDF5 files have high performance I/O.

Note: Test set is typically simulated separately with different model parameters from the training and validation sets.
Therefore, the HDF5 files either contain train and val groups, or test group only.

The script creates the following HDF5 files:
- QFinder_feature_train_val.h5
- QFinder_feature_test.h5 (if split_mode is test)
- RASFinder_feature_train_val.h5
- RASFinder_feature_test.h5 (if split_mode is test)
- FFinder_feature_train_val.h5
- FFinder_feature_test.h5 (if split_mode is test)

Usage:
    # For train/val split by iteration number:
    python data_preparation/package_features.py \
        --qfinder_dir ./extracted_features_train_val/QFinder \
        --rasfinder_dir ./extracted_features_train_val/RASFinder \
        --ffinder_dir ./extracted_features_train_val/FFinder \
        --output_dir ./hdf5_features \
        --split_mode iteration \
        --split_threshold 160

    # For train/val split by random:
    python data_preparation/package_features.py \
        --qfinder_dir ./extracted_features_train_val/QFinder \
        --rasfinder_dir ./extracted_features_train_val/RASFinder \
        --ffinder_dir ./extracted_features_train_val/FFinder \
        --output_dir ./hdf5_features \
        --split_mode random \
        --train_ratio 0.8
        --seed 723

    # For test set (no splitting):
    python data_preparation/package_features.py \
        --qfinder_dir ./extracted_features_test/QFinder \
        --rasfinder_dir ./extracted_features_test/RASFinder \
        --ffinder_dir ./extracted_features_test/FFinder \
        --output_dir ./hdf5_features \
        --split_mode test
"""

import argparse
import re
import h5py
import numpy as np
from pathlib import Path
import random
import time


def extract_iteration(filename):
    """
    Extract iteration number from filename.
    
    Filename format: {label}({n_taxa}){model}[+F][+I{...}][+G4{...}][{n_sites}]_{iteration}
    
    Args:
        filename: Filename string (with or without extension)
    
    Returns:
        Iteration number or None if not found
    """
    match = re.search(r'_(\d+)(?:\.(?:npy|npz))?$', filename)
    if match:
        return int(match.group(1))
    return None


def split_by_iteration(files, threshold):
    """
    Split files into train/val sets based on iteration number.
    
    Files with iteration < threshold go to train, >= threshold go to val.
    
    Args:
        files: List of file names
        threshold: Iteration threshold for splitting
    
    Returns:
        Tuple of (train_files, val_files), each containing file names
    """
    train_files = []
    val_files = []
    
    for filename in files:
        iteration = extract_iteration(filename)
        if iteration is not None:
            if iteration < threshold:
                train_files.append(filename)
            else:
                val_files.append(filename)
        else:
            train_files.append(filename)
    
    return train_files, val_files


def split_randomly(files, train_ratio=0.8, seed=723):
    """
    Split files randomly into train/val sets.
    
    Args:
        files: List of file names
        train_ratio: Proportion for training set
        seed: Random seed
    
    Returns:
        Tuple of (train_files, val_files), each containing file names
    """
    random.seed(seed)
    files_shuffled = files.copy()
    random.shuffle(files_shuffled)
    
    n_total = len(files_shuffled)
    n_train = int(n_total * train_ratio)
    
    train_files = files_shuffled[:n_train]
    val_files = files_shuffled[n_train:]
    
    return train_files, val_files


def package_to_hdf5(feature_dir, output_path, groups_dict, key_extractor):
    """
    Package features into a single HDF5 file.
    
    Args:
        feature_dir: Directory containing feature files
        output_path: Path to output HDF5 file
        groups_dict: Dictionary mapping group names to file lists
                     e.g., {"train": [files...], "val": [files...]} or {"test": [files...]}
        key_extractor: Function to extract HDF5 key from filename
    
    Returns:
        Dictionary mapping group names to number of samples saved
    """
    feature_dir = Path(feature_dir)
    saved_counts = {}
    
    with h5py.File(output_path, "w") as h5:
        for group_name, file_list in groups_dict.items():
            group = h5.create_group(group_name)
            saved_count = 0
            
            for filename in file_list:
                file_path = feature_dir / filename
                if file_path.exists():
                    arr = np.load(file_path)
                    key = key_extractor(filename)
                    group.create_dataset(key, data=arr)
                    saved_count += 1
            
            saved_counts[group_name] = saved_count
    
    return saved_counts


def find_rasfinder_files(rasfinder_dir):
    """
    Find all RASFinder .npz files in directory.
    
    Args:
        rasfinder_dir: Directory containing RASFinder features
    
    Returns:
        Dictionary mapping base_name to filename
    """
    rasfinder_dir = Path(rasfinder_dir)
    all_files = [f for f in rasfinder_dir.glob("*.npz")]
    
    return {f.stem: f.name for f in all_files}


def package_qfinder_features(qfinder_dir, output_path, groups_dict):
    """
    Package QFinder features into HDF5 file.
    
    Args:
        qfinder_dir: Directory containing QFinder .npy files
        output_path: Path to output HDF5 file
        groups_dict: Dictionary mapping group names to file lists
    """
    def get_key(filename):
        return Path(filename).stem
    
    saved_counts = package_to_hdf5(qfinder_dir, output_path, groups_dict, get_key)
    
    for group_name, count in saved_counts.items():
        print(f"Saved {count} samples to {group_name} group")


def package_rasfinder_features(rasfinder_dir, output_path, groups_dict, file_dict):
    """
    Package RASFinder features (sitewise and summary) into a single HDF5 file.
    
    Each sample is stored as a group containing both sitewise and summary features.
    
    Args:
        rasfinder_dir: Directory containing RASFinder .npz files
        output_path: Path to output HDF5 file
        groups_dict: Dictionary mapping group names to base name lists
        file_dict: Dictionary mapping base_name to filename (to avoid redundant file lookup)
    """
    rasfinder_dir = Path(rasfinder_dir)
    saved_counts = {}
    
    with h5py.File(output_path, "w") as h5:
        for group_name, base_list in groups_dict.items():
            split_group = h5.create_group(group_name)
            saved_count = 0
            
            for base_name in base_list:
                if base_name not in file_dict:
                    continue
                
                npz_path = rasfinder_dir / file_dict[base_name]
                if npz_path.exists():
                    npz_data = np.load(npz_path)
                    sample_group = split_group.create_group(base_name)
                    sample_group.create_dataset("sitewise", data=npz_data['sitewise'])
                    sample_group.create_dataset("summary", data=npz_data['summary'])
                    saved_count += 1
            
            saved_counts[group_name] = saved_count
    
    for group_name, count in saved_counts.items():
        print(f"Saved {count} samples to {group_name} group")


def package_ffinder_features(ffinder_dir, output_path, groups_dict):
    """
    Package FFinder features into HDF5 file.
    
    Args:
        ffinder_dir: Directory containing FFinder .npy files
        output_path: Path to output HDF5 file
        groups_dict: Dictionary mapping group names to file lists
    """
    def get_key(filename):
        return Path(filename).stem
    
    saved_counts = package_to_hdf5(ffinder_dir, output_path, groups_dict, get_key)
    
    for group_name, count in saved_counts.items():
        print(f"Saved {count} samples to {group_name} group")


def split_files_by_mode(files, split_mode, split_threshold=None, train_ratio=0.8, seed=723):
    """
    Split files into train/val/test groups based on split mode.
    
    Args:
        files: List of file names or base names
        split_mode: "iteration", "random", or "test"
        split_threshold: Iteration threshold (for "iteration" mode)
        train_ratio: Proportion for training set (for "random" mode)
        seed: Random seed (for "random" mode)
    
    Returns:
        Dictionary mapping group names to file lists
    """
    if split_mode == "test":
        return {"test": files}
    elif split_mode == "iteration":
        train_files, val_files = split_by_iteration(files, split_threshold)
        return {"train": train_files, "val": val_files}
    else:
        train_files, val_files = split_randomly(files, train_ratio, seed)
        return {"train": train_files, "val": val_files}


def package_features(
    qfinder_dir,
    rasfinder_dir,
    ffinder_dir,
    output_dir,
    split_mode="iteration",
    split_threshold=160,
    train_ratio=0.8,
    seed=723
):
    """
    Package features from feature files into HDF5 format.
    
    Args:
        qfinder_dir: Directory with QFinder features
        rasfinder_dir: Directory with RASFinder features
        ffinder_dir: Directory with FFinder features
        output_dir: Directory to save HDF5 files
        split_mode: "iteration" (split by iteration number) or "random" (random split) or "test" (test set only)
        split_threshold: Iteration threshold for train/val split (used when split_mode="iteration")
        train_ratio: Proportion for training set (used when split_mode="random")
        seed: Random seed (used when split_mode="random")
    """
    qfinder_dir = Path(qfinder_dir)
    rasfinder_dir = Path(rasfinder_dir)
    ffinder_dir = Path(ffinder_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output file suffix based on split mode
    if split_mode == "test":
        file_suffix = "_test"
    else:
        file_suffix = "_train_val"
    
    # Process QFinder features
    
    qfinder_files = sorted([f.name for f in qfinder_dir.glob("*.npy")])
    print(f"Found {len(qfinder_files)} QFinder feature files")

    qfinder_groups = split_files_by_mode(qfinder_files, split_mode, split_threshold, train_ratio, seed)
    if split_mode == "test":
        print("Processing as test set (no splitting)")
    elif split_mode == "iteration":
        print(f"Split by iteration (threshold={split_threshold}): "
              f"{len(qfinder_groups['train'])} train, {len(qfinder_groups['val'])} val")
    else:
        print(f"Random split (train_ratio={train_ratio}): "
              f"{len(qfinder_groups['train'])} train, {len(qfinder_groups['val'])} val")
    
    package_qfinder_features(qfinder_dir, output_dir / f"QFinder_feature{file_suffix}.h5", qfinder_groups)
   
    # Process RASFinder features
    # Find files once and reuse the dictionary
    rasfinder_file_dict = find_rasfinder_files(rasfinder_dir)
    rasfinder_bases = sorted(rasfinder_file_dict.keys())
    rasfinder_groups = split_files_by_mode(rasfinder_bases, split_mode, split_threshold, train_ratio, seed)
    package_rasfinder_features(rasfinder_dir, output_dir / f"RASFinder_feature{file_suffix}.h5", rasfinder_groups, rasfinder_file_dict)
    
    # Process FFinder features
    ffinder_files = sorted([f.name for f in ffinder_dir.glob("*.npy")])
    ffinder_groups = split_files_by_mode(ffinder_files, split_mode, split_threshold, train_ratio, seed)
    package_ffinder_features(ffinder_dir, output_dir / f"FFinder_feature{file_suffix}.h5", ffinder_groups)

def main():
    parser = argparse.ArgumentParser(
        description="Package features from .npy files to HDF5 format"
    )
    parser.add_argument(
        "--qfinder_dir",
        type=str,
        required=True,
        help="Directory with QFinder features"
    )
    parser.add_argument(
        "--rasfinder_dir",
        type=str,
        required=True,
        help="Directory with RASFinder features"
    )
    parser.add_argument(
        "--ffinder_dir",
        type=str,
        required=True,
        help="Directory with FFinder features"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hdf5_features",
        help="Directory to save HDF5 files (default: ./hdf5_features)"
    )
    parser.add_argument(
        "--split_mode",
        type=str,
        choices=["iteration", "random", "test"],
        default="iteration",
        help="Split mode: 'iteration' (by iteration number), 'random' (random split), or 'test' (test set only)"
    )
    parser.add_argument(
        "--split_threshold",
        type=int,
        default=160,
        help="Iteration threshold for train/val split (used when split_mode='iteration', default: 160)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Proportion for training set (used when split_mode='random', default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=723,
        help="Random seed for splitting (default: 723)"
    )
    
    args = parser.parse_args()
    start_total = time.perf_counter()
    
    package_features(
        qfinder_dir=args.qfinder_dir,
        rasfinder_dir=args.rasfinder_dir,
        ffinder_dir=args.ffinder_dir,
        output_dir=args.output_dir,
        split_mode=args.split_mode,
        split_threshold=args.split_threshold,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    end_total = time.perf_counter()
    total_time = end_total - start_total

    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
