"""
PyTorch Dataset classes for loading features from HDF5 files.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union
from pathlib import Path


class QFinderDataset(Dataset):
    """
    Dataset for QFinder model: loads QFinder features from HDF5 files.
    
    Each sample contains:
    - Features: (625, 440) array reshaped to (440, 25, 25)
    - Label: Substitution model class (0-6)
    
    The label is extracted from the HDF5 key's first character.
    """
    
    def __init__(
        self,
        h5_paths: Union[str, List[str]],
        group_name: str = "train"
    ):
        """
        Args:
            h5_paths: Path(s) to HDF5 file(s). If multiple paths provided, samples from all files will be combined.
            group_name: Name of the HDF5 group (e.g., "train", "val", "test")
        """
        if isinstance(h5_paths, str):
            h5_paths = [h5_paths]
        
        self.h5_paths = [Path(p) for p in h5_paths]
        self.group_name = group_name
        
        self.h5_files = []
        self.h5_groups = []
        self.keys = []
        self.index_to_key = []
        
        for file_idx, h5_path in enumerate(self.h5_paths):
            h5_file = h5py.File(h5_path, "r")
            group = h5_file[group_name]
            keys = sorted(group.keys())
            
            self.h5_files.append(h5_file)
            self.h5_groups.append(group)
            self.keys.extend(keys)
            
            for key in keys:
                self.index_to_key.append((file_idx, key))  # file_idx: index of the h5 file, key: key of the sample
    

    def __len__(self) -> int:
        return len(self.keys)
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, key = self.index_to_key[idx]
        data = self.h5_groups[file_idx][key][:].astype(np.float32)
        # Reshape: (625, 440) -> (25, 25, 440) -> (440, 25, 25)
        feature = torch.from_numpy(data.reshape(25, 25, 440)).permute(2, 0, 1)
        # Extract label from key's first character, [0-6]
        label = int(key[0])
        
        return feature, torch.tensor(label, dtype=torch.long)
    

    def __del__(self):
        """Closes all HDF5 files when dataset is deleted."""
        for h5_file in self.h5_files:
            try:
                h5_file.close()
            except Exception:
                pass


def map_label_to_multi(label):
    """
    Convert RHAS label to dual-head multi-label [has_G, has_I]
    """

    if isinstance(label, bytes):
        label = label.decode()

    label = str(label)

    if label == "None":
        return [0, 0]

    elif label == "+G":
        return [1, 0]

    elif label == "+I":
        return [0, 1]

    elif label == "+G+I":
        return [1, 1]

    else:
        raise ValueError(f"Unknown label: {label}")


class RASFinderDataset(Dataset):
    """
    Dataset for RASFinder model: loads RASFinder features from HDF5 files.
    
    Each sample contains:
    - Sitewise features: (n_sites, 23) array
    - Summary features: (10,) array
    - Label: RHAS model class (0-3)
    
    The HDF5 file structure:
    - Each sample is a group containing 'sitewise' and 'summary' datasets
    - Groups are organized under train/val/test splits
    
    The label is extracted from the HDF5 key's first character.
    """
    
    def __init__(
        self,
        h5_paths: Union[str, List[str]],
        group_name: str = "train"
    ):
        """
        Args:
            h5_paths: Path(s) to HDF5 file(s). If multiple paths provided, samples from all files will be combined.
            group_name: Name of the HDF5 group (e.g., "train", "val", "test")
        """
        if isinstance(h5_paths, str):
            h5_paths = [h5_paths]
        
        self.h5_paths = [Path(p) for p in h5_paths]
        self.group_name = group_name
        
        self.h5_files = []
        self.h5_groups = []
        self.keys = []
        self.index_to_key = []
        
        for file_idx, h5_path in enumerate(self.h5_paths):
            h5_file = h5py.File(h5_path, 'r')
            group = h5_file[group_name]
            keys = sorted(group.keys())
            
            # Verify each sample has both sitewise and summary features
            for key in keys:
                sample_group = group[key]
                if 'sitewise' not in sample_group:
                    raise ValueError(
                        f"Sample '{key}' in {h5_path} missing sitewise feature"
                    )
                if 'summary' not in sample_group:
                    raise ValueError(
                        f"Sample '{key}' in {h5_path} missing summary feature"
                    )
            
            self.h5_files.append(h5_file)
            self.h5_groups.append(group)
            self.keys.extend(keys)
            
            for key in keys:
                self.index_to_key.append((file_idx, key))
    

    def __len__(self) -> int:
        return len(self.keys)
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_idx, key = self.index_to_key[idx]
        sample_group = self.h5_groups[file_idx][key]
        sitewise_feature = torch.from_numpy(sample_group['sitewise'][:].astype(np.float32))  # (n_sites, 23)
        summary_feature = torch.from_numpy(sample_group['summary'][:].astype(np.float32))    # (10,)
        # Extract label from key's first character, [0-3]
        label = int(key[0])
        
        return sitewise_feature, summary_feature, torch.tensor(label, dtype=torch.long)
    
    
    def __del__(self):
        for h5_file in self.h5_files:
            try:
                h5_file.close()
            except Exception:
                pass


def collate_fn_rasfinder(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for RASFinderDataset to handle variable-length sequences.
    
    Pads sitewise features to the same length within a batch.
    
    Args:
        batch: List of (sitewise_features, summary_features, label) tuples
    
    Returns:
        Tuple of:
        - sitewise_features: Padded tensor of shape (B, max_n_sites, 23)
        - summary_features: Tensor of shape (B, 10)
        - lengths: Tensor of shape (B,) with actual sequence lengths
        - labels: Tensor of shape (B,)
    """

    sitewise_features, summary_features, labels = zip(*batch)
    
    # Get actual lengths
    lengths = torch.tensor([f.shape[0] for f in sitewise_features], dtype=torch.long)
    max_length = lengths.max().item()
    
    # Pad sitewise features
    padded_sitewise = []
    for feature in sitewise_features:
        n_sites = feature.shape[0]
        if n_sites < max_length:
            # Pad with zeros
            padding = torch.zeros(max_length - n_sites, feature.shape[1], dtype=feature.dtype)
            padded = torch.cat([feature, padding], dim=0)
        else:
            padded = feature
        padded_sitewise.append(padded)
    
    # Stack into batch tensors
    sitewise_batch = torch.stack(padded_sitewise)  # (B, max_n_sites, 23)
    summary_batch = torch.stack(summary_features)  # (B, 10)
    labels_batch = torch.stack(labels)            # (B,)
    
    return sitewise_batch, summary_batch, lengths, labels_batch
